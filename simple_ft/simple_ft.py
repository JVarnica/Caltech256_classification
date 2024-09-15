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
from linear_probe.model_eval import get_dali_loader
from models.bs_model_wrapper import BaseTimmWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(messages)s')

def get_exp_config(dataset_name):
    config_module = importlib.import_module(f'configs.{dataset_name}_config')
    return getattr(config_module, f'{dataset_name.upper()}_CONFIG')

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        inputs, labels = data[0]['image'], data[0]['label'].squeeze(-1).long()
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
    accuracy = 100 * correct / total

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
    accuracy = 100 * correct / total 

    return avg_loss, accuracy


def train_and_evaluate(model, train_loader, val_loader, criterion, device, num_epochs, config):
    best_val_acc = 0
    best_model_state = None
    train_losses, train_accs, val_losses, val_accs, epoch_times = [], [], [], [], []
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    min_improvement = config['min_improvement']
    early_stop_patience= config['early_stop_patience']
    scheduler_patience = config['scheduler_patience']
    epochs_no_imprv = 0 

    optimizer = AdamW(model.get_trainable_params(), lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=scheduler_patience, verbose=True)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        new_stage, previous_stage_state = model.adaptive_unfreeze(epoch, num_epochs, best_val_acc)
        if new_stage:
            optimizer = AdamW(model.get_trainable_params(), lr, weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=scheduler_patience, verbose=True)
            epochs_no_imprv = 0
        
        model.train()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        model.eval()
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        epoch_time = time.time() - epoch_start_time
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)

        if val_acc > best_val_acc + min_improvement:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f'{config['model_name']}_{config['dataset_name']}_best.pth')
            epochs_no_imprv = 0
        else:
            epochs_no_imprv += 1

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Epoch Time: {epoch_time:2f}s'
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        #Logic of early stopping
        if epochs_no_imprv >= early_stop_patience:
            if model.unfreeze_state['current_stage'] < model.unfreeze_state['total_stages'] - 1:
                if model.unfreeze_state['stage_history'] and model.unfreeze_state['stage_history'][-1][1] == 'performance':
                    logging.info(f"No Improvement for {early_stop_patience}, after stage change. Early Stopping Activated!!")
                    break
                else:
                    new_stage,_ = model.adaptive_unfreeze(epoch, num_epochs, best_val_acc)
                    if new_stage:
                        optimizer = AdamW(model.get_trainable_params(), lr, weight_decay)
                        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=scheduler_patience, verbose=True)
                        epochs_no_imprv = 0
                        logging.info(f"Moving onto stage {model.unfreeze_state['current_stage']+ 1}")
                    else:
                        logging.info(f"Patience Reached and Fully Unfrozen!!")
                        break
            else:
                logging.info(f"No more Stages!!")
                break
    total_time = time.time() - start_time()
    logging.info(f"{model.model_name} total training time: {total_time:.4f} seconds")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, train_accs, val_losses, val_accs, best_val_acc, epoch_times, total_time

def run_experiment(config, model_name):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running experiment for {model_name}")
    
    model = BaseTimmWrapper(config['model_name'], config['num_classes'], 
                            freeze_mode=config['freeze_mode'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    train_loader = get_dali_loader(config['train_dir'], config['batch_size'], 
                                   config['num_threads'], 0, model.get_config(), is_training=True)
    val_loader = get_dali_loader(config['val_dir'], config['batch_size'], 
                                 config['num_threads'], 0, model.get_config(), is_training=False)
    
    train_losses, train_accs, val_losses, val_accs, best_val_acc, epoch_times, total_time= train_and_evaluate(
        model, train_loader, val_loader, criterion, device, config['num_epochs'], config
    )
    
    result = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'epoch_time': epoch_times,
        'total_time': total_time
    }
    
    # Save results
    save_results(result, config['results_dir'], model_name)

    return result

def save_results(result, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to CSV
    csv_path = os.path.join(results_dir, f"{model_name}_metrics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Epoch Time'])
        for i in range(len(result['train_losses'])):
            writer.writerow([i+1, result['train_losses'][i], result['train_accs'][i], 
                             result['val_losses'][i], result['val_accs'][i], result['epoch_time']])
    
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

def main():
    parser = argparse.ArgumentParser(description='Run fine-tuning on dataset')
    parser.add_argument('dataset', choices=['caltech256', 'cifar100'], help='Dataset to use')
    parser.add_argument('model', help='Model to fine-tune')
    args = parser.parse_args()

    config = get_exp_config(args.dataset)
    
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

