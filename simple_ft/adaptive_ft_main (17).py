
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
from timm.data.mixup import MixUp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

def setup_logging(dataset_name, model_name, results_dir, experiment_name, ft_strategy):
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{experiment_name}_{ft_strategy}_{dataset_name}_{model_name}.log')
    
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
    config_module = importlib.import_module(f'configs.data_aug_{dataset_name}_config')
    return getattr(config_module, f'DATA_AUG_{dataset_name.upper()}_CONFIG')

def get_dali_loader():
    # Lazy import loader for tests.
    module = importlib.import_module('data.data_aug')
    return module.get_dali_loader

def get_base_model():
    module = importlib.import_module('models.bs_model_wrapper')
    return module.BaseTimmWrapper

def train_epoch(model, train_loader, optimizer, criterion, device, scaler, config):

    mixup_fn = None
    if config['data_aug_config']['loader'] == 'strong':
      mixup_params = config['data_aug_config']['transforms']['Mix_Cut']['params']
      mixup_fn = MixUp( mixup_alpha= config['data_aug_config']['transforms']['Mix_Cut']['Params'],
          mixup_alpha=mixup_params['mixup_alpha'],
          cutmix_alpha=mixup_params['cutmix_alpha'],
          prob=mixup_params['prob'],
          switch_prob=mixup_params['switch_prob'],
          mode='batch',
          label_smoothing=mixup_params['label_smoothing']
  )

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        inputs, labels  = data[0]['image'], data[0]['label'].squeeze(-1).long()
        inputs, labels = inputs.to(device), labels.to(device)

        if mixup_fn is not None:
            mixed_images, mixed_labels = mixup_fn(inputs, labels)

        optimizer.zero_grad()

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels) #how can i check with made up labels

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #evaluate on original labels MUST
        with torch.no_grad():
          probs = F.softmax(outputs, dim=1)
          conf, pred = torch.max(probs, dim=1)
          conf_sum += conf.mean().item()
          total += labels.size(0)
          correct += pred.eq(labels).sum().item()

        total_loss += loss.item()
        

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * (correct / total)
    avg_confidence = conf_sum / len(train_loader)

    return avg_loss, accuracy, avg_confidence

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
            probs = F.softmax(outputs, dim=1)
            conf, predicted = torch.max(probs, dim=1)
            conf_sum += conf.mean().item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * (correct / total)
    avg_confidence = conf_sum / len(val_loader)

    return avg_loss, accuracy, avg_confidence

def train_and_evaluate(model, train_loader, val_loader, criterion, device, num_epochs, config, callback=None):

    """ Training process combines different fine-tuning strategies"""
    best_val_acc = 0
    model_checkpoint = None
    train_losses, train_accs, train_confs, val_losses, val_accs, val_confs, epoch_times = [], [], [], [], [], [], []
    epochs_no_improve = 0
    weight_decay = config['weight_decay']
    min_improvement = config['min_improvement'] 
    early_stop_patience = config['early_stop_patience']
    plateau_patience = config['plateau_patience']
    base_lr = config['base_lr']
    max_adapt = config['max_adapt']
    adapt_count = 0
    plateau_counter = 0
    pending_stg_change = False

    if model.ft_strategy == 'full':
        optimizer = AdamW(model.get_trainable_params(), lr=base_lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif model.ft_strategy == 'gradual':
        trainable_params = model.get_trainable_params()
        if not trainable_params:
            raise ValueError("No trainable parameters found at initialization!")
        optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)
        scheduler = None
    elif model.ft_strategy == 'discriminative':
        optimizer = AdamW([{'params': group['params'], 'lr': base_lr * group['lr_mult']} 
                         for group in model.param_groups], weight_decay=weight_decay)
        scheduler = None
    elif model.ft_strategy == 'complete':
        optimizer = AdamW(model.get_trainable_params(), lr=base_lr, weight_decay=weight_decay)
        scheduler = None


    scaler = GradScaler('cuda')
    scaler._enabled = True
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Update stage progress and learning rates at epoch start
        model.update_stage_progress(optimizer, epoch)
        
        if model.ft_strategy in ['gradual', 'complete']:
          if pending_stg_change:
              trainable_params = model.get_trainable_params()
              if model.ft_strategy == "complete":
                # Create optimizer with coordinated learning rates for each stage
                  try:
                        optimizer = model.create_stage_optimizer(
                            config['base_lr'], 
                            epoch,
                            config['weight_decay'])
                        logging.info(f"Created new optimizer for stage {model.current_stage} with "
                            f"{sum(p.numel() for group in optimizer.param_groups for p in group['params'])}"
                            f"trainable parameters")
                  except ValueError as e:
                        logging.error(f"Failed to create optimizer: {str(e)}")
                        raise ValueError
              else: 
                  optimizer = AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)

              epochs_no_improve = 0
              plateau_counter = 0
              pending_stg_change = False

        model.train()
        train_loss, train_acc, train_conf = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        model.eval()
        val_loss, val_acc, val_conf = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_confs.append(train_conf)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_confs.append(val_conf)
        epoch_times.append(epoch_time)

        if val_acc > best_val_acc + min_improvement:
            best_val_acc = val_acc
            model_checkpoint = model.state_dict().copy()
            epochs_no_improve = 0
            plateau_counter = 0
        else:
            epochs_no_improve += 1
            plateau_counter += 1
        
        curr_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Epoch Time: {epoch_time:.2f}s, '
                     f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                     f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val_Conf: {val_conf} '
                     f'Learning rate: {curr_lr:.6f}'
        )

        if plateau_counter >= plateau_patience: # global max val acc didnt increase for plateau. (if not there continuous stg chngs as val_acc is updated)
          logging.info(f"No improvement for {plateau_patience} epochs. Early stopping.")
          break
        
        if adapt_count >= max_adapt: # just another safeguard so cannot chnage lr or stgs more than i want which is 5.
          logging.info(f"Maximum number of adaptions. Stop Trainning!!")
          break

        # stage progressions
        if epochs_no_improve >= early_stop_patience and adapt_count < max_adapt: # Patience reached 
            if model.ft_strategy in ['gradual', 'complete']:
              stage_change = model._gradual_unfreeze(force_unfreeze=True)
              if stage_change:
                pending_stg_change = True
                adapt_count += 1
                
            elif model.ft_strategy == 'discriminative':
              if model._adjust_lr():
                optimizer = AdamW([{'params': group['params'], 'lr': base_lr * group['lr_mult']} for group in model.param_groups], weight_decay=weight_decay)
                logging.info(f"Reduced learning rates. New rates: {[group['lr'] for group in optimizer.param_groups]}")
                adapt_count += 1
                epochs_no_improve = 0

        elif model.ft_strategy in  ['gradual', 'complete']:
              stage_change = model._gradual_unfreeze(force_unfreeze=False)
              if stage_change:
                  pending_stg_change = True
                  adapt_count += 1
                  epochs_no_improve = 0
                  

        if train_acc > 99.99:
            logging.info("Training accuracy reached 99.99%. Stopping training.")
            break

        if callback:
            callback(epoch, model, optimizer, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]['lr'])

    # End of training time
    total_time = time.time() - start_time
    logging.info(f"{model.model_name} total training time: {total_time:.4f} seconds")

    #Save best model
    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint)

    return {
        'train_losses': train_losses, 
        'train_accs': train_accs, 
        'train_confs': train_confs,
        'val_losses': val_losses, 
        'val_accs': val_accs, 
        'val_confs': val_confs,
        'best_val_acc': best_val_acc, 
        'epoch_times': epoch_times, 
        'total_time': total_time, 
        'best_model_state': model_checkpoint,
        
    }

def save_results(result, results_dir, dataset_name, model_name, ft_strategy, experiment_name): # not dynamic yet
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to CSV
    csv_path = os.path.join(results_dir, f"{experiment_name}_{dataset_name}_{model_name}_{ft_strategy}_metrics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Train Conf', 'Val Loss', 'Val Acc', 'Val Conf', 'Epoch Time'])
        for i in range(len(result['train_losses'])):
            writer.writerow([i+1, result['train_losses'][i], result['train_accs'][i], result['train_confs'][i], 
                             result['val_losses'][i], result['val_accs'][i], result['val_confs'][i] ,result['epoch_times'][i]])
    
    
    # Plot and save learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(result['train_losses'], label='Train')
    plt.plot(result['val_losses'], label='Validation')
    plt.title(f'data_aug_{dataset_name} - {model_name} Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(result['train_accs'], label='Train')
    plt.plot(result['val_accs'], label='Validation')
    plt.title(f'Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{experiment_name}_{dataset_name}_{model_name}_{ft_strategy}_learning_curves.png"))
    logging.info(f"{experiment_name}_{model_name} results saved!!")
    plt.close()

def save_model(model, dataset_name, models_dir, ft_strategy, experiment_name):
    """Save Model with stage-wise parameter organization"""
    model_name = model.model_name
    num_classes = model.num_classes
    
    # Create mapping of parameter IDs to their stage names
    param_to_stage = {}
    for group in model.param_groups:
        stage_name = group['name']
        # Use parameter IDs instead of direct comparison
        for param in group['params']:
            param_to_stage[id(param)] = stage_name
    
    # Get all parameter names and organize them by stage
    stage_states = {}
    state_dict = model.state_dict()
    
    # Initialize stage dictionaries
    for group in model.param_groups:
        stage_name = group['name']
        stage_states[stage_name] = {
            'parameters': {},
            'statistics': {},
            'learning_rate_multiplier': group.get('lr_mult', 1.0)
        }
    
    # Organize parameters by stage
    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id in param_to_stage:
            stage_name = param_to_stage[param_id]
            stage_states[stage_name]['parameters'][name] = state_dict[name]
    
    # Calculate statistics for each stage
    for stage_name, stage_info in stage_states.items():
        param_tensors = [t for t in stage_info['parameters'].values() 
                        if isinstance(t, torch.Tensor)]
        
        if param_tensors:
            try:
                stage_info['statistics'] = {
                    'param_count': sum(p.numel() for p in param_tensors),
                    'mean_magnitude': torch.mean(torch.stack([p.abs().mean() 
                        for p in param_tensors])).item(),
                    'std_magnitude': torch.mean(torch.stack([p.std() 
                        for p in param_tensors])).item(),
                    'sparsity': torch.mean(torch.stack([(p == 0).float().mean() 
                        for p in param_tensors])).item()
                }
            except RuntimeError as e:
                logging.warning(f"Could not compute all statistics for {stage_name}: {str(e)}")
                stage_info['statistics'] = {
                    'param_count': sum(p.numel() for p in param_tensors)
                }
    
    save_dict = {
        'model_info': {
            'model_name': model_name,
            'num_classes': num_classes,
            'dataset_name': dataset_name,
            'ft_strategy': ft_strategy
        },
        'complete_state': state_dict,
        'stage_states': stage_states,
        'training_config': {
            'total_stages': model.total_stages,
            'current_stage': model.current_stage,
            'epochs_in_current_stage': model.epochs_in_current_stage,
        }
    }
    
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 
                            f'{experiment_name}_{dataset_name}_{model_name}_{ft_strategy}_best.pth')
    torch.save(save_dict, save_path)
    logging.info(f"Model saved with layer states: {save_path}")


def run_experiment(config, model_name, ft_strategy ,callback=None):

    dataset_name = config['dataset_name']
    experiment_name = config['experiment_name']
    data_aug_config = config['data_aug_config'] 
    models_dir = config['models_dir']
    results_dir = config['results_dir']

    logging.info(f"Starting {experiment_name} for {model_name} on {dataset_name} while using {ft_strategy}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Running experiment for {model_name}")

    BaseTimmWrapper = get_base_model()
    model = BaseTimmWrapper(model_name, config['num_classes'], 
                            ft_strategy=ft_strategy, 
                            head_epochs=config['head_epochs'], 
                            stage_epochs=config['stage_epochs'],
                            warmup_epochs=config.get('warmup_epochs')
                            )
                            
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    dali_loader = get_dali_loader()
    train_loader = dali_loader(data_dir=config['train_dir'], batch_size=config['batch_size'], 
                                   num_threads=config['num_threads'], device_id=0, model_config=model.get_config(), data_aug_config=data_aug_config, is_training=True)
    val_loader = dali_loader(data_dir=config['val_dir'], batch_size=config['batch_size'], 
                                 num_threads=config['num_threads'], device_id=0, model_config=model.get_config(), data_aug_config={'loader': 'none'}, is_training=False)
    
    results = train_and_evaluate(model, train_loader, val_loader, criterion, device, config['num_epochs'], config, callback)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    save_results(results, results_dir, dataset_name, model_name, ft_strategy, experiment_name)
    save_model(model, dataset_name, models_dir, ft_strategy, experiment_name)

    return results

def main():
    parser = argparse.ArgumentParser(description='Run diff fine-tuning experiments with different data augmentation strats on cifar100 & caltech256')
    parser.add_argument('dataset', choices=['caltech256', 'cifar100', 'mock_data'], help='Dataset to use')
    parser.add_argument('model', help=' Model architecture to fine-tune (e.g. resnet50, vit_base)')
    parser.add_argument('experiment', choices=['no_aug', 'basic_aug', 'strong_aug', 'randaug'], 
                          help= 'Data augmentation experiment you are doing')
    parser.add_argument('--ft_strategy', choices=['full', 'gradual', 'discriminative', 'complete'], default='full', help='Fine-tuning strategy')
    args = parser.parse_args()

    config = get_exp_config(args.dataset)
    model_config = next((m for m in config['model_list'] if m['model_name'] == args.model), None)

    if model_config is None:
        available_models = [m['model_name'] for m in config['defaults']['model_list']]
        raise ValueError(f"Model {args.model} not found. Available models: {available_models}")
    
    # Update config with model-specific parameters
    if args.experiment not in config['experiments']:
        raise ValueError(f"Unknown experiment!!!" )
    
    
    experiment_config = config['experiments'][args.experiment]

    final_config = {
      **config['defaults'],
      **experiment_config,
      'model_config': model_config,
      'train_dir': config['train_dir'],
      'val_dir': config['val_dir'],
      'monitoring': config['monitoring'],
      'dataset_name': args.dataset,
      'experiment_name': args.experiment,
      'ft_strategy': args.ft_strategy
    }

    setup_logging(args.dataset, args.model, final_config['results_dir'], final_config['experiment_name'], final_config['ft_strategy'],)
    
    result = run_experiment(final_config, args.model, final_config['ft_strategy'])
    
    # Print summary
    logging.info(f"\nExperiment Result Summary for {args.model} on {args.dataset}:")
    logging.info(f"Best Validation Accuracy = {result['best_val_acc']:.2f}%")
    logging.info(f"Total Training Time = {result['total_time']:.2f} seconds")

if __name__ == '__main__':
    main()

