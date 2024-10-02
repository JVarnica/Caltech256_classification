import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch
import torch
import os
from simple_ft.simple_ft import run_experiment, get_exp_config

def test_run_experiment_with_mock_data():
    # Get mock data config
    config = get_exp_config('mock_data')

    # Update config for quick testing
    config.update({
        'num_epochs': 5,
        'batch_size': 4,
        'stage_lrs': [0.001, 0.0005, 0.0001, 0.00001],
        'early_stop_patience': 2,
        'min_improvement': 0.0001
    })

    # List of models to test
    models_to_test = [
        'resnet50.a1_in1k',
        'vit_base_patch16_224.orig_in21k_ft_in1k',
        'pvt_v2_b3.in1k',
        'regnety_040.pycls_in1k'
    ]

    for model_name in models_to_test:
        print(f"Testing {model_name}...")

        model_config = next((model for model in config['model_list'] if model['model_name'] == model_name), None)
        if model_config is None:
            raise ValueError(f"Model {model_name} not found in configuration for dataset mock_data")
        
        # Create experiment config
        experiment_config = config.copy()
        experiment_config.update(model_config)
        experiment_config['model_name'] = model_name
        experiment_config['freeze_mode'] = model_config.get('freeze_mode', 'gradual')
        experiment_config['unfreeze_epochs'] = [1, 2, 3]

        callback_data = []

        def mock_callback(epoch, model, optimizer ,train_loss, train_acc, val_loss, val_acc):
            callback_data.append({
            'epoch': epoch,
            'stage': model.unfreeze_state['current_stage'],
            'lr': optimizer.param_groups[0]['lr'],
            'optimizer': optimizer,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        
        # Run the experiment
        with patch('torch.save'), patch('simple_ft.simple_ft.save_results'):
            result = run_experiment(config, model_name)

        # Perform checks
        assert result is not None, f"Result is None for {model_name}"
        assert 'best_val_acc' in result, f"No best_val_acc in result for {model_name}"
        assert 'train_losses' in result, f"No train_losses in result for {model_name}"
        assert 'val_losses' in result, f"No val_losses in result for {model_name}"
        assert 'train_accs' in result, f"No train_accs in result for {model_name}"
        assert 'val_accs' in result, f"No val_accs in result for {model_name}"
        assert 'epoch_times' in result, f"No epoch_times in result for {model_name}"
        assert 'total_time' in result, f"No total_time in result for {model_name}"

        assert len(result['train_losses']) == config['num_epochs'], f"Incorrect number of epochs for {model_name}"

        assert all(0 <= loss <= 10 for loss in result['train_losses']), f"Unreasonable train losses for {model_name}"
        assert all(0 <= acc <= 100 for acc in result['train_accs']), f"Unreasonable train accuracies for {model_name}"
        assert all(0 <= loss <= 10 for loss in result['val_losses']), f"Unreasonable validation losses for {model_name}"
        assert all(0 <= acc <= 100 for acc in result['val_accs']), f"Unreasonable validation accuracies for {model_name}"

        assert max(result['val_accs']) > result['val_accs'][0], f"No improvement in validation accuracy for {model_name}"
        
        stages = [data['stage'] for data in callback_data]
        learning_rates = [data['lr'] for data in callback_data]

        assert len(set(stages)) > 1, f"No stage changes detected for {model_name}"
        assert stages[-1] >= len(experiment_config['unfreeze_epochs']), f"Not all stages were reached for {model_name}"
        assert len(set(learning_rates)) > 1, f"No learning rate changes detected for {model_name}"
        assert learning_rates[0] == experiment_config['stage_lrs'][0], f"Initial learning rate incorrect for {model_name}"
        
        # Check if the number of unique learning rates matches the number of stages
        assert len(set(learning_rates)) == len(set(stages)), f"Mismatch between stage changes and learning rate updates for {model_name}"

        # Check if callback data matches result data
        for i, data in enumerate(callback_data):
            assert abs(data['train_loss'] - result['train_losses'][i]) < 1e-6, f"Train loss mismatch at epoch {i} for {model_name}"
            assert abs(data['train_acc'] - result['train_accs'][i]) < 1e-6, f"Train accuracy mismatch at epoch {i} for {model_name}"
            assert abs(data['val_loss'] - result['val_losses'][i]) < 1e-6, f"Validation loss mismatch at epoch {i} for {model_name}"
            assert abs(data['val_acc'] - result['val_accs'][i]) < 1e-6, f"Validation accuracy mismatch at epoch {i} for {model_name}"

        print(f"All checks passed for {model_name}")
        print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
        print(f"Total training time: {result['total_time']:.2f} seconds")
        print("--------------------")

if __name__ == '__main__':
    test_run_experiment_with_mock_data()