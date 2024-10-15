import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch
import torch
import os
import logging
from simple_ft.simple_ft import run_experiment, get_exp_config

class TestRunExperiment(unittest.TestCase):
    def setUp(self):
        # Get mock data config
        self.config = get_exp_config('mock_data')

        # Update config for quick testing
        self.config.update({
            'num_epochs': 5,
            'batch_size': 4,
            'early_stop_patience': 2,
            'min_improvement': 0.0001,
            'head_epochs': 2,
            'stage_epochs': 3
        })

        # List of models to test
        self.models_to_test = [
            'resnet50.a1_in1k',
            'vit_base_patch16_224.orig_in21k_ft_in1k',
            'pvt_v2_b3.in1k',
            'regnety_040.pycls_in1k'
        ]
    @patch('torch.save')
    @patch('simple_ft.simple_ft.save_results')
    def test_run_experiment(self, mock_save_results, mock_torch_save):
        for model_name in self.models_to_test:
            with self.subTest(model=model_name):
                print(f"Testing {model_name}...")

                log_dir = 'test_logs'
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f'{model_name}_test.log')
                logging.basicConfig(filename=log_file, level=logging.INFO, 
                                    format='%(asctime)s - %(levelname)s - %(message)s',
                                    force=True)

                model_config = next((model for model in self.config['model_list'] if model['model_name'] == model_name), None)
                self.assertIsNotNone(model_config, f"Model {model_name} not found in configuration for dataset mock_data")
                
                experiment_config = self.config.copy()
                experiment_config.update(model_config)

                callback_data = []

                def callback(epoch, model, optimizer ,train_loss, train_acc, val_loss, val_acc, lr):
                    callback_data.append({
                    'epoch': epoch,
                    'stage': model.unfreeze_state['current_stage'],
                    'lr': lr,
                    'optimizer': optimizer,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                })
                result = run_experiment(experiment_config, model_name, callback)

                # Perform checks
                self.assertIsNotNone(result, f"Result is None for {model_name}")
                self.assertIn('best_val_acc', result, f"No best_val_acc in result for {model_name}")
                self.assertIn('train_losses', result, f"No train_losses in result for {model_name}")
                self.assertIn('val_losses', result, f"No val_losses in result for {model_name}")
                self.assertIn('train_accs', result, f"No train_accs in result for {model_name}")
                self.assertIn('val_accs', result, f"No val_accs in result for {model_name}")
                self.assertIn('epoch_times', result, f"No epoch_times in result for {model_name}")
                self.assertIn('total_time', result, f"No total_time in result for {model_name}")

                self.assertEqual(len(result['train_losses']), experiment_config['num_epochs'], f"Incorrect number of epochs for {model_name}")

                self.assertTrue(all(0 <= loss <= 10 for loss in result['train_losses']), f"Unreasonable train losses for {model_name}")
                self.assertTrue(all(0 <= acc <= 100 for acc in result['train_accs']), f"Unreasonable train accuracies for {model_name}")
                self.assertTrue(all(0 <= loss <= 10 for loss in result['val_losses']), f"Unreasonable validation losses for {model_name}")
                self.assertTrue(all(0 <= acc <= 100 for acc in result['val_accs']), f"Unreasonable validation accuracies for {model_name}")

                # Check if there's any improvement in validation accuracy
                self.assertGreater(max(result['val_accs']), result['val_accs'][0], f"No improvement in validation accuracy for {model_name}")

                stages = [data['stage'] for data in callback_data]
                learning_rates = [data['lr'] for data in callback_data]

                self.assertGreater(len(set(stages)), 1, f"No stage changes detected for {model_name}")
                self.assertGreater(len(set(learning_rates)), 1, f"No learning rate changes detected for {model_name}")
                self.assertEqual(learning_rates[0], experiment_config['stage_lrs'][0], f"Initial learning rate incorrect for {model_name}")

                # Check if the number of unique learning rates matches the number of stages
                self.assertEqual(len(set(learning_rates)), len(set(stages)), f"Mismatch between stage changes and learning rate updates for {model_name}")

                print(f"All checks passed for {model_name}")
                print(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
                print(f"Total training time: {result['total_time']:.2f} seconds")
                print(f"Stage progression: {stages}")
                print(f"Learning rate progression: {learning_rates}")
                print("--------------------")

                for handler in logging.root.handlers[:]:
                     logging.root.removeHandler(handler)

if __name__ == '__main__':
        unittest.main()