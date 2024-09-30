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

def handle_new_stage(model, optimizer, scheduler, stage_best_val_acc, stage_checkpoint, stage_results, epoch, lr, weight_decay, scheduler_patience):
        if stage_checkpoint is not None:
            stage_results.append({
                'stage': model.unfreeze_state['current_stage'] - 1,
                'val_acc': stage_best_val_acc,
                'model_state': stage_checkpoint,
                'epoch': epoch - 1
            })
            logging.info(f"Saved optimal results/model for stage{model.unfreeze_state['current_stage']-1}")

        stage_best_val_acc = 0
        stage_checkpoint = None
        optimizer = AdamW(model.get_trainable_params(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=scheduler_patience, verbose=True)
        return optimizer, scheduler, stage_best_val_acc, stage_checkpoint


def train_and_evaluate(model, train_loader, val_loader, criterion, device, num_epochs, config, callback=None):
    best_val_acc = 0
    stage_best_val_acc = 0
    best_model_state = None
    stage_checkpoint = None
    train_losses, train_accs, val_losses, val_accs, epoch_times = [], [], [], [], []
    stage_results = []
    epochs_no_improve = 0
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    min_improvement = config['min_improvement'] 
    early_stop_patience= config['early_stop_patience']
    scheduler_patience = config['scheduler_patience']
    

    optimizer = AdamW(model.get_trainable_params(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=scheduler_patience, verbose=True)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        if callback:
            callback(epoch, model, optimizer, scheduler, train_loss, train_acc, val_loss, val_acc)
       
        new_stage  = model.adaptive_unfreeze(epoch, best_val_acc, False)
        if new_stage:
            logging.info(f"Epoch {epoch+1}: Scheduled stage transition")
            optimizer, scheduler, stage_best_val_acc, stage_checkpoint = handle_new_stage(
                model, optimizer, scheduler, stage_best_val_acc, stage_checkpoint, stage_results,
                epoch, lr, weight_decay, scheduler_patience
            )
            epochs_no_improve = 0
        
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
            best_model_checkpoint = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if val_acc > stage_best_val_acc:
            stage_best_val_acc = val_acc
            stage_checkpoint = model.state_dict().copy()
        

        logging.info(f'Epoch {epoch+1}/{num_epochs}, Epoch Time: {epoch_time:2f}s, '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        #Logic of early stopping

        if epochs_no_improve >= early_stop_patience:
            if not model.unfreeze_state['stage_history'] or  model.unfreeze_state['stage_history'][-1][1] != 'performance': #empty OR epoch CONTINUE
                new_stage = model.adaptive_unfreeze(epoch, best_val_acc, True)
                if new_stage:
                    logging.info(f"No improv for {early_stop_patience} epochs. Moving to next stage {model.unfreeze_state['current_stage']}")
                    optimizer, scheduler, stage_best_val_acc, stage_checkpoint = handle_new_stage(
                        model, optimizer, scheduler, stage_best_val_acc, stage_checkpoint, stage_results,
                        epoch, lr, weight_decay, scheduler_patience
                    )
                    epochs_no_improve = 0
                else:
                    logging.info(f"No more stages available. Early Stopping")
                    break
            else:
                logging.info(f"No Improvement for {early_stop_patience} after performance-based chnage last time. Early stopping")
                break
            
    if stage_checkpoint is not None:
        stage_results.append({
            'stage': model.unfreeze_state['current_stage'],
            'val_acc': stage_best_val_acc,
            'model_state': stage_checkpoint,
            'epoch': epoch
        })
        logging.info(f"Saved final stage checkpoint, {model.unfreeze_state['current_stage']}")

    total_time = time.time() - start_time()
    logging.info(f"{model.model_name} total training time: {total_time:.4f} seconds")

    if best_model_checkpoint is not None:
        model.load_state_dict(best_model_state)

    return {
        'train_losses':train_losses, 
        'train_accs':train_accs, 
        'val_losses': val_losses, 
        'val_accs': val_accs, 
        'best_val_acc': best_val_acc, 
        'epoch_times': epoch_times, 
        'total_time': total_time, 
        'stage_results': stage_results,
        'best_model_state': best_model_state
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running experiment for {model_name}")

    BaseTimmWrapper = get_base_model()
    model = BaseTimmWrapper(config['model_name'], config['num_classes'], 
                            freeze_mode=config['freeze_mode'], unfreeze_epochs=config['unfreeze_epochs'])
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
    spef_model_dir = os.path.join(models_dir, model_name)
    os.makedirs(spef_model_dir, exist_ok=True)

    best_model_path = os.path.join(spef_model_dir,  f'{config["model_name"]}_overall_best.pth')
    torch.save(results['best_model_state'], best_model_path)
    logging.info(f"Saved best {model_name} to {best_model_path}")

    #  Save stage-wise best models and print results
    for stage_result in results['stage_results']:
        stage_best_path = os.path.join(spef_model_dir, f'{config["dataset_name"]}_stage_{stage_result["stage"]}_best.pth')
        torch.save(stage_result['model_state'], stage_best_path)
        logging.info(f"Stage {stage_result['stage']} Best Val Acc: {stage_result['val_acc']:.4f} "
                     f"(Epoch {stage_result['epoch']}), Saved to {stage_best_path} ")
    
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

