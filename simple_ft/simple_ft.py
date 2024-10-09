import os
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        inputs, labels  = data[0]['image'], data[0]['label'].squeeze(-1).long()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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
    early_stop_patience= config['early_stop_patience']
    head_epochs = config['head_epochs']
    stage_epochs = config['stage_epochs']
    

    optimizer = AdamW(model.get_trainable_params(), lr=config['stage_lrs'][0], weight_decay=weight_decay, betas=(0.9, 0.999))

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        model.eval()
        val_loss, val_acc = validate(model, val_loader, criterion, device)

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
        else:
            epochs_no_improve += 1

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Epoch Time: {epoch_time:2f}s, '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        #Logic of early stopping

        if callback:
            callback(epoch, model, optimizer, train_loss, train_acc, val_loss, val_acc)

        patience_reached = epochs_no_improve >= early_stop_patience
        new_stage = model.adaptive_unfreeze(patience_reached)

        if new_stage == "early_stop":
            logging.info(f"No Improvement for {early_stop_patience} epochs, after performance-based chnaged last time. Early Stopping")
            break
        elif new_stage:
            current_stage = model.unfreeze_state['current_stage']
            new_lr = config['stage_lrs'][current_stage] if current_stage < len(config['stage_lrs']) else config['stage_lrs'][-1]
            optimizer = AdamW(model.get_trainable_params(), lr=new_lr, weight_decay=weight_decay, betas=(0.9, 0.999))
            epochs_no_improve = 0
            if model.unfreeze_state['stage_history'][-1][1] == 'performance': 
                logging.info(f"Performance-based transition to stage {current_stage}, with learning rate: {new_lr}")
            else:
                logging.info(f"Epoch-Based transition to stage {current_stage} with learning rate: {new_lr} ")
        elif patience_reached:
            if model.unfreeze_state['current_stage'] < model.unfreeze_state['total_stages'] -1:
                logging.info(f"Forced transition to next stage as patience reached")
                continue
            else:
                logging.info(f"Patience Reached and no more stages available. Early Stopping!!")
                break
                

    total_time = time.time() - start_time
    logging.info(f"{model.model_name} total training time: {total_time:.4f} seconds")

    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint)

    return {
        'train_losses':train_losses, 
        'train_accs':train_accs, 
        'val_losses': val_losses, 
        'val_accs': val_accs, 
        'best_val_acc': best_val_acc, 
        'epoch_times': epoch_times, 
        'total_time': total_time, 
        'best_model_state': model_checkpoint
    }

def save_results(result, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to CSV
    csv_path = os.path.join(results_dir, f"{model_name}_metrics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Epoch Time'])
        for i in range(len(result['train_losses'])):
            writer.writerow([i+1, result['train_losses'][i], result['train_accs'][i], 
                             result['val_losses'][i], result['val_accs'][i], result['epoch_times']])
    
    # Plot and save learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(result['train_losses'], label='Train')
    plt.plot(result['val_losses'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(result['train_accs'], label='Train')
    plt.plot(result['val_accs'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{model_name}_learning_curves.png"))
    logging.info(f"{model_name} results saved!!")
    plt.close()

def run_experiment(config, model_name):

    logging.info("Config received in run_experiment:", config)
    logging.info("Model name received in run_experiment:", model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running experiment for {model_name}")

    BaseTimmWrapper = get_base_model()
    model = BaseTimmWrapper(model_name, config['num_classes'], 
                            freeze_mode=config.get('freeze_mode'), head_epochs=config['head_epochs'], stage_epochs=config['stage_epochs'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    dali_loader = get_dali_loader()
    train_loader = dali_loader(config['train_dir'], config['batch_size'], 
                                   config['num_threads'], 0, model.get_config(), is_training=True)
    val_loader = dali_loader(config['val_dir'], config['batch_size'], 
                                 config['num_threads'], 0, model.get_config(), is_training=False)
    
    results = train_and_evaluate(
        model, train_loader, val_loader, criterion, device, config['num_epochs'], config
    )
    
    base_dir = os.path.dirname(os.path.dirname(config['results_dir']))
    models_dir = os.path.join(base_dir, 'models')

    best_model_path = os.path.join(models_dir,  f'{config["model_name"]}_overall_best.pth')
    torch.save(results['best_model_state'], best_model_path)
    logging.info(f"Saved best {model_name} to {best_model_path}")
    
    # Save results
    save_results(results, config['results_dir'], model_name)

    return results

def main():
    parser = argparse.ArgumentParser(description='Run fine-tuning on dataset')
    parser.add_argument('dataset', choices=['caltech256', 'cifar100', 'mock_data'], help='Dataset to use')
    parser.add_argument('model', help='Model to fine-tune')
    args = parser.parse_args()

    config = get_exp_config(args.dataset)
    config['dataset_name'] = args.dataset
    
    model_config = next((m for m in config['model_list'] if m['model_name'] == args.model), None)

    if model_config is None:
        raise ValueError(f"Model {args.model} not found in configuration for dataset {args.dataset}")
    # Update config with model-specific parameters
    experiment_config = config.copy()
    experiment_config.update(model_config)
    
    logging.info(f"Starting experiment for model: {args.model}")
    
    result = run_experiment(experiment_config, args.model)
    
    # Print summary
    logging.info(f"\nExperiment Result Summary for {args.model}:")
    logging.info(f"Best Validation Accuracy = {result['best_val_acc']:.2f}%")
    logging.info(f"Total Training Time = {result['total_time']:.2f} seconds")

if __name__ == '__main__':
    main()

