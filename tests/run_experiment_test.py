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

        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(self.test_dir, 'test_logs')
        os.makedirs(self.log_dir, exist_ok=True)

    def setup_logging(self, model_name):
        log_file = os.path.join(self.log_dir, f'{model_name}_test.log')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            force=True)

    @patch('torch.save')
    @patch('simple_ft.simple_ft.save_results')
    def test_run_experiment(self, mock_save_results, mock_torch_save):
        for model_config in self.config['model_list']:
            with self.subTest(model=model_config['model_name']):
                self.setup_logging(model_config['model_name'])
                logging.info(f"Testing {model_config['model_name']} with {model_config['ft_strategy']} strategy...")
                experiment_config = self.config.copy()
                experiment_config.update(model_config)

                callback_data = []

                def callback(epoch, model, optimizer ,train_loss, train_acc, val_loss, val_acc, lr):
                    callback_data.append({
                    'epoch': epoch,
                    'stage': getattr(model, 'current_stage', 0),
                    'lr': lr,
                    'optimizer': optimizer,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                })
                result = run_experiment(experiment_config, model_config['model_name'], callback)

                logging.info(f"All checks passed for {model_config['model_name']}")
                logging.info(f"Best validation accuracy: {result['best_val_acc']:.2f}%")
                logging.info(f"Total training time: {result['total_time']:.2f} seconds")
                logging.info(f"Stage progression: {[data['stage'] for data in callback_data]}")
                logging.info(f"Learning rate progression: {[data['lr'] for data in callback_data]}")
                logging.info("--------------------")

    def run_assertions(self, result, config, model_name, callback_data):

            self.assertIsNotNone(result, f"Result is None for {model_name}")
            self.assertIn('best_val_acc', result, f"No best_val_acc in result for {model_name}")
            self.assertIn('train_losses', result, f"No train_losses in result for {model_name}")
            self.assertIn('val_losses', result, f"No val_losses in result for {model_name}")
            self.assertIn('train_accs', result, f"No train_accs in result for {model_name}")
            self.assertIn('val_accs', result, f"No val_accs in result for {model_name}")
            self.assertIn('epoch_times', result, f"No epoch_times in result for {model_name}")
            self.assertIn('total_time', result, f"No total_time in result for {model_name}")

            self.assertEqual(len(result['train_losses']), config['num_epochs'], f"Incorrect number of epochs for {model_name}")

            self.assertTrue(all(0 <= loss <= 10 for loss in result['train_losses']), f"Unreasonable train losses for {model_name}")
            self.assertTrue(all(0 <= acc <= 100 for acc in result['train_accs']), f"Unreasonable train accuracies for {model_name}")
            self.assertTrue(all(0 <= loss <= 10 for loss in result['val_losses']), f"Unreasonable validation losses for {model_name}")
            self.assertTrue(all(0 <= acc <= 100 for acc in result['val_accs']), f"Unreasonable validation accuracies for {model_name}")

            # Check if there's any improvement in validation accuracy
            self.assertGreater(max(result['val_accs']), result['val_accs'][0], f"No improvement in validation accuracy for {model_name}")

            stages = [data['stage'] for data in callback_data]
            learning_rates = [data['lr'] for data in callback_data]

            if config['freeze_mode'] == 'gradual':
                self.assertGreater(len(set(stages)), 1, f"No stage changes detected for {model_name}")
            elif config['freeze_mode'] == 'discriminative':
                self.assertEqual(len(set(stages)), 1, f"Unexpected stage changes for discriminative strategy in {model_name}")
            elif config['freeze_mode'] == 'full':
                self.assertEqual(len(set(stages)), 1, f"Unexpected stage changes for full fine-tuning in {model_name}")

            self.assertEqual(learning_rates[0], config['base_lr'], f"Initial learning rate incorrect for {model_name}")
           
if __name__ == '__main__':
        unittest.main()