import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock
import torch
from simple_ft.simple_ft import train_and_evaluate


class MockCallback:
    def __init__(self):
        self.calls = []
    
    def __call__(self, epoch, model, optimizer ,train_loss, train_acc, val_loss, val_acc):
        self.calls.append({
            'epoch': epoch,
            'model_stage': model.unfreeze_state['current_stage'],
            'optimizer': optimizer,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MagicMock()
        self.model.model_name = "TestModel"
        self.model.get_trainable_params.return_value  = [torch.nn.Parameter(torch.randn(1)) for _ in range(5)]
        self.model.state_dict.return_value = {}
        self.criterion = MagicMock()
        self.config = {
            'num_epochs': 20,
            'early_stop_patience': 5,
            'min_improvement': 0.001,
            'stage_lrs': [0.001, 0.0005, 0.0001, 0.00001],
            'weight_decay': 1e-5,
            'unfreeze_epochs': [2, 4]
        }
    @patch('simple_ft.simple_ft.train_epoch')
    @patch('simple_ft.simple_ft.validate')
    @patch('torch.optim.AdamW')
    # Simulate early stopping. 1 stage change, 2nd needs to break not change.
    def test_early_stopping(self, mock_AdamW, mock_validate, mock_train_epoch):
        
        mock_train_epoch.return_value = (0.5, 80.0)
        val_losses = [0.4, 0.39, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.37, 0.34, 0.33, 0.37, 0.37, 0.38, 0.35, 0.35]
        val_accs = [ 82.0, 82.5, 85.0, 83.0, 82.0, 81.0, 82.0, 82.5, 82.5, 86.0, 86.5, 84.5, 84.5 , 84.0, 84.5, 84.5]
                       #                 #                              #                                          #
        mock_validate.side_effect = list(zip(val_losses, val_accs))
        mock_callback = MockCallback()

        class DynamicMockOptimizer:
            def __init__(self, *args, **kwargs):
                self.param_groups = [{'lr': kwargs['lr']}]

        mock_AdamW.side_effect = DynamicMockOptimizer

        def adaptive_unfreeze_side_effect(epoch, patience_reached=False):
            if patience_reached:
                self.model.unfreeze_state['stage_history'].append((self.model.unfreeze_state['current_stage'] + 1, 'performance'))
                self.model.unfreeze_state['current_stage'] += 1
                return True
            elif epoch in self.config['unfreeze_epochs']:
                self.model.unfreeze_state['stage_history'].append((self.model.unfreeze_state['current_stage'] + 1, 'epoch'))
                self.model.unfreeze_state['current_stage'] += 1
                return True
            return False

        self.model.adaptive_unfreeze.side_effect = adaptive_unfreeze_side_effect
          
        self.model.unfreeze_state = {'stage_history': [], 'current_stage': 0}

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                     self.config['num_epochs'], self.config, callback=mock_callback)
        
        self.assertEqual(len(results['val_accs']), 16)
        # Check if stage transitions occurred correctly
        expected_stages = [(1, 'epoch'), (2, 'epoch'), (3, 'performance')]
        self.assertEqual(self.model.unfreeze_state['stage_history'], expected_stages)
        self.assertEqual(len(mock_callback.calls), 16)

        for i, call in enumerate(mock_callback.calls):
            self.assertEqual(call['epoch'], i)
            self.assertEqual(call['train_loss'], 0.5)
            self.assertEqual(call['train_acc'], 80.0)
            self.assertEqual(call['val_loss'], val_losses[i])
            self.assertEqual(call['val_acc'], val_accs[i])

            if i < 2:
                self.assertEqual(call['lr'], self.config['stage_lrs'][0])
            if i == 2:
                self.assertEqual(call['lr'], self.config['stage_lrs'][1])
            elif i == 4:
                self.assertEqual(call['lr'], self.config['stage_lrs'][2])
            elif i == 9: 
                self.assertEqual(call['lr'], self.config['stage_lrs'][3])
            elif i > 9:
                self.assertEqual(call['lr'], self.config['stage_lrs'][3])

if __name__ == '__main__':
    unittest.main()