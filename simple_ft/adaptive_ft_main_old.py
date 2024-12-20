
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import importlib
import csv
import time
import logging
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR


def setup_logging(dataset_name, model_name, results_dir):
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{dataset_name}_{model_name}.log')
    
    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

def get_exp_config(dataset_name):
    config_module = importlib.import_module(f'configs.simple_ft_{dataset_name}_config')
    return getattr(config_module, f'SIMPLE_FT_{dataset_name.upper()}_CONFIG')

def get_dali_loader():
    # Lazy import loader for tests.
    module = importlib.import_module('linear_probe.model_eval')
    return module.get_dali_loader

def get_base_model():
    module = importlib.import_module('models.bs_model_wrapper')
    return module.BaseTimmWrapper

def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        inputs, labels  = data[0]['image'], data[0]['label'].squeeze(-1).long()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * (correct / total)

    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0]['image'], data[0]['label'].squeeze(-1).long()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * (correct / total)

    return avg_loss, accuracy

def train_and_evaluate(model, train_loader, val_loader, criterion, device, num_epochs, config, callback=None):
    best_val_acc = 0
    model_checkpoint = None
    train_losses, train_accs, val_losses, val_accs, epoch_times = [], [], [], [], []
    epochs_no_improve = 0
    weight_decay = config['weight_decay']
    min_improvement = config['min_improvement'] 
    early_stop_patience = config['early_stop_patience']
    plateau_patience = config['plateau_patience']
    base_lr = config['base_lr']
    max_adapt = config['max_adapt']
    adapt_count = 0

    if model.ft_strategy == 'full':
        optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif model.ft_strategy == 'gradual':
        # Start with only the head parameters
        optimizer = AdamW(model.get_trainable_params(), lr= base_lr, weight_decay=weight_decay)
        scheduler = None
    elif model.ft_strategy == 'discriminative':
        optimizer = AdamW([{'params': group['params'], 'lr': base_lr * group['lr_mult']} for group in model.param_groups], weight_decay=weight_decay)
        scheduler = None

    scaler = GradScaler()
    scaler._enabled = True

    start_time = time.time()
    pending_stg_change = False

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        if pending_stg_change and model.ft_strategy == 'gradual':
            logging.info(f"Updating optimizer for stage {model.current_stage}")
            trainable_params = list(model.get_trainable_params())
            logging.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
            optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)
            epochs_no_improve = 0
            pending_stg_change = False
      
        model.train()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        model.eval()
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)

        if val_acc > best_val_acc + min_improvement:
            best_val_acc = val_acc
            model_checkpoint = model.state_dict().copy()
            epochs_no_improve = 0
            plateau_counter = 0
        else:
            epochs_no_improve += 1
            plateau_counter += 1

        if model.ft_strategy == 'discriminative':
            lr_string = ", ".join([f"Group {i}: {group['lr']:.2e}" for i, group in enumerate(optimizer.param_groups)])
        else:
            lr_string = f"{optimizer.param_groups[0]['lr']:.2e}"

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Epoch Time: {epoch_time:.2f}s, '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                     f'Learning rate: {lr_string}, '
                     f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        if plateau_counter >= plateau_patience:
          logging.info(f"No improvement for {plateau_patience} epochs. Early stopping.")
          break
        
        if adapt_count >= max_adapt:
          logging.info(f"Maximum number of adaptions. Stop Trainning!!")
          break

        if epochs_no_improve >= early_stop_patience and adapt_count < max_adapt:
            if model.ft_strategy == 'gradual':
              stage_change = model.adaptive_unfreeze(force_unfreeze=True)
              if stage_change:
                pending_stg_change = True
                adapt_count += 1
                epochs_no_improve = 0
            elif model.ft_strategy == 'discriminative':
              changed = model.adaptive_unfreeze(force_unfreeze=True)
              if changed:
                optimizer = AdamW([{'params': group['params'], 'lr': base_lr * group['lr_mult']} for group in model.param_groups], weight_decay=weight_decay)
                logging.info(f"Reduced learning rates. New rates: {[group['lr'] for group in optimizer.param_groups]}")
                adapt_count += 1
                epochs_no_improve = 0
        elif model.ft_strategy == 'gradual':
              stage_change = model.adaptive_unfreeze(force_unfreeze=False)
              if stage_change:
                  pending_stg_change = True
                  adapt_count += 1
                  epochs_no_improve = 0

        if train_acc > 99.98:
            logging.info("Training accuracy reached 99.99%. Stopping training.")
            break

        if callback:
            callback(epoch, model, optimizer, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr'])

    total_time = time.time() - start_time
    logging.info(f"{model.model_name} total training time: {total_time:.4f} seconds")

    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint)

    return {
        'train_losses': train_losses, 
        'train_accs': train_accs, 
        'val_losses': val_losses, 
        'val_accs': val_accs, 
        'best_val_acc': best_val_acc, 
        'epoch_times': epoch_times, 
        'total_time': total_time, 
        'best_model_state': model_checkpoint
    }

def save_results(result, results_dir, dataset_name, model_name, ft_strategy):
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to CSV
    csv_path = os.path.join(results_dir, f"{ft_strategy}_{dataset_name}_{model_name}_metrics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Epoch Time'])
        for i in range(len(result['train_losses'])):
            writer.writerow([i+1, result['train_losses'][i], result['train_accs'][i], 
                             result['val_losses'][i], result['val_accs'][i], result['epoch_times'][i]])
    
    # Plot and save learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(result['train_losses'], label='Train')
    plt.plot(result['val_losses'], label='Validation')
    plt.title(f'{ft_strategy}_{dataset_name} - {model_name} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(result['train_accs'], label='Train')
    plt.plot(result['val_accs'], label='Validation')
    plt.title(f'{dataset_name} - {model_name} Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{dataset_name}_{model_name}_learning_curves.png"))
    logging.info(f"{dataset_name}_{model_name} results saved!!")
    plt.close()

def save_model(model, dataset_name, models_dir, ft_strategy):
    model_name = model.model_name
    num_classes = model.num_classes

    save_dict = {
        'state_dict': model.state_dict(),
        'model_name': model_name,
        'num_classes': num_classes,
        'dataset_name': dataset_name
    }
    save_path = os.path.join(models_dir, f'{ft_strategy}_{dataset_name}_{model_name}_best.pth')
    torch.save(save_dict, save_path)
    logging.info(f"Model saved {save_path}")


def run_experiment(config, model_name, ft_strategy ,callback=None):

    dataset_name = config['dataset_name']

    logging.info("Starting experiment for {model_name} on {dataset_name} using {ft_strategy}")
    logging.info("Configuration: {config}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running experiment for {model_name}")

    BaseTimmWrapper = get_base_model()
    model = BaseTimmWrapper(model_name, config['num_classes'], 
                            ft_strategy=ft_strategy, 
                            head_epochs=config['head_epochs'], 
                            stage_epochs=config['stage_epochs'])
                            
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    dali_loader = get_dali_loader()
    train_loader = dali_loader(config['train_dir'], config['batch_size'], 
                                   config['num_threads'], 0, model.get_config(), is_training=True)
    val_loader = dali_loader(config['val_dir'], config['batch_size'], 
                                 config['num_threads'], 0, model.get_config(), is_training=False)
    
    results = train_and_evaluate(model, train_loader, val_loader, criterion, device, config['num_epochs'], config, callback)
    
    models_dir = config['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    # Save results
    save_results(results, config['results_dir'], dataset_name, model_name, ft_strategy)
    save_model(model, dataset_name, models_dir, ft_strategy)

    return results

def main():
    parser = argparse.ArgumentParser(description='Run fine-tuning on dataset')
    parser.add_argument('dataset', choices=['caltech256', 'cifar100', 'mock_data'], help='Dataset to use')
    parser.add_argument('model', help='Model to fine-tune')
    parser.add_argument('--ft_strategy', choices=['full', 'gradual', 'discriminative'], default=None, help='Fine-tuning strategy')
    args = parser.parse_args()

    config = get_exp_config(args.dataset)
    config['dataset_name'] = args.dataset
    
    model_config = next((m for m in config['model_list'] if m['model_name'] == args.model), None)


    if model_config is None:
        raise ValueError(f"Model {args.model} not found in configuration for dataset {args.dataset}")
    
    # Update config with model-specific parameters
    experiment_config = config.copy()
    experiment_config.update(model_config)
    if args.ft_strategy:
      experiment_config['ft_strategy'] = args.ft_strategy

    logging.info(f"Fine-tuning strategy: {experiment_config['ft_strategy']}")

    setup_logging(args.dataset, args.model, experiment_config['results_dir'])
    
    result = run_experiment(experiment_config, args.model, experiment_config['ft_strategy'])
    
    # Print summary
    logging.info(f"\nExperiment Result Summary for {args.model} on {args.dataset}:")
    logging.info(f"Best Validation Accuracy = {result['best_val_acc']:.2f}%")
    logging.info(f"Total Training Time = {result['total_time']:.2f} seconds")

if __name__ == '__main__':
    main()

