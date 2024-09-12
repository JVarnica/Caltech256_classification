import os
import argparse
import importlib
import csv
import time
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from linear_probe.model_eval import get_dali_loader
from models.bs_model_wrapper import BaseTimmWrapper

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
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    patience = config['patience']
    min_improvement = config['min_improvement']
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    early_stop_partience= config['early_stop_patience']
    scheduler_patience = config['scheduler_patience']
    epochs_no_imprv = 0 

    param_groups = model.get_param_groups()
    optimizer = AdamW(param_groups, lr, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=scheduler_patience, verbose=True)

    for epoch in range(num_epochs):
        stop_training, previous_stage_state = model.adaptive_unfreeze(epoch, num_epochs, best_val_acc)
        if stop_training:
            print("Early Stopping Triggred")
            break

        optimizer.param_groups = model.get_param_groups()

        for param_group in optimizer.param_groups:
            if param_group['params'][0].requires_grad and 'lr' not in param_groups:
                param_group['lr'] = config['learning_rate']
        
        model.train()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        model.eval()
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc + min_improvement:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f'{config['model_name']}_{config['dataset_name']}_best.pth')
            epochs_no_imprv = 0
        else:
            epochs_no_imprv += 1

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if epochs_no_imprv >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break

        if previous_stage_state:
            # Evaluate performance after moving to a new stage
            evaluation_epochs = min(patience, num_epochs - epoch - 1)
            best_new_stage_accuracy = val_acc

            for eval_epoch in range(evaluation_epochs):
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                scheduler.step(val_acc)

                if val_acc > best_new_stage_accuracy:
                    best_new_stage_accuracy = val_acc

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()

            # If the new stage didn't improve performance, revert to the previous stage
            if best_new_stage_accuracy <= model.unfreeze_state['best_performance']:
                print(f"New stage did not improve performance. Reverting to previous stage.")
                model.load_state_dict(previous_stage_state)
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, train_accs, val_losses, val_accs, best_val_acc

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
    
    train_losses, train_accs, val_losses, val_accs, best_val_acc = train_and_evaluate(
        model, train_loader, val_loader, criterion, device, config['num_epochs'], config
    )
    
    result = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
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
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
        for i in range(len(result['train_losses'])):
            writer.writerow([i+1, result['train_losses'][i], result['train_accs'][i], 
                             result['val_losses'][i], result['val_accs'][i]])
    
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
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run fine-tuning on dataset')
    parser.add_argument('dataset', choices=['caltech256', 'cifar100'], help='Dataset to use')
    parser.add_argument('model', help='Model to fine-tune')
    args = parser.parse_args()

    config = get_exp_config(args.dataset)
    
    result = run_experiment(config, args.model)
    
    # Print summary
    print("\nExperiment Result Summary:")
    print(f"{result['model_name']}: Best Validation Accuracy = {result['best_val_acc']:.2f}%")

if __name__ == '__main__':
    main()

