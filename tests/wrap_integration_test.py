import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock
import torch
from simple_ft.simple_ft import train_and_evaluate


class MockAdamW(torch.optim.Optimizer): #Mock object doesn't mock optimizer properly
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super(MockAdamW, self).__init__(params, defaults)
    
    def step(self, closure=None):
        pass

class TestEarlyStopping(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MagicMock()
        self.model.model_name = "TestModel"
        self.mock_params = [
            torch.nn.Parameter(torch.randn(1)),
            torch.nn.Parameter(torch.randn(1)),
            torch.nn.Parameter(torch.randn(1)),
            torch.nn.Parameter(torch.randn(1)),
            torch.nn.Parameter(torch.randn(1))
        ]
        self.model.get_trainable_params.return_value = self.mock_params
        self.model.state_dict.return_value = {}
        self.criterion = MagicMock()
        self.config = {
            'num_epochs': 20,
            'early_stop_patience': 5,
            'min_improvement': 0.001,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'scheduler_patience': 3
        }
    @patch('simple_ft.simple_ft.train_epoch')
    @patch('simple_ft.simple_ft.validate')
    @patch('simple_ft.simple_ft.handle_new_stage')
    @patch('torch.optim.AdamW', MockAdamW)
    @patch('torch.optim.lr_scheduler.ReduceLROnPlateau', autospec=True)
    def test_early_stopping(self, mock_scheduler, mock_handle_new_stage, mock_validate, mock_train_epoch):
        
        mock_train_epoch.return_value = [0.5, 80.0]
        mock_validate.side_effect = [
            (0.4, 82.0), (0.39, 82.5), (0.38, 85.0),  # Improving
            (0.38, 83.0), (0.38, 82.0), (0.38, 81.0), (0.38, 82.0), (0.38, 83.0),  # Stagnating (5 epochs)
            (0.37, 85.5), (0.36, 86.0), (0.35, 86.5),  # Improving aft stage transition
            (0.35, 84.5), (0.35, 84.5), (0.35, 84.5), (0.35, 84.5), (0.35, 84.5)  # Stagnating again (5 epochs)
        ]
        self.model.adaptive_unfreeze.side_effect = [
            False, False, False, 
            False, False, False, False, 
            True,
            False, False, False, 
            False, False, False,
            True,
        ]
        self.model.unfreeze_state = {'stage_history': [], 'current_stage': 0}

        mock_optimizer = MockAdamW(self.model.get_trainable_params())
        mock_scheduler_instance = MagicMock()
        mock_scheduler.return_value = mock_scheduler_instance

        def side_effect_handle_new_stage(*args, **kwargs):
            self.model.unfreeze_state['stage_history'].append((1, 'performance'))
            return (mock_optimizer, mock_scheduler_instance, 0, None)
        
        mock_handle_new_stage.side_effect = side_effect_handle_new_stage

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                     self.config['num_epochs'], self.config)
        
        self.assertEqual(len(results['val_accs']), 15)
        self.model.adaptive_unfreeze.assert_any_call(7, self.config['num_epochs'], 83.0, True)
        mock_handle_new_stage.assert_called_once()
        self.assertEqual(self.model.unfreeze_state['stage_history'], [(1, 'performance')])

if __name__ == '__main__':
    unittest.main()