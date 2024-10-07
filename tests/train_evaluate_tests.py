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

class DynamicMockOptimizer:
    def __init__(self, *args, **kwargs):
        self.param_groups = [{'lr': kwargs['lr']}]

class TestTrain_Evaluate(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MagicMock()
        self.model.model_name = "TestModel"
        self.model.get_trainable_params.return_value  = [torch.nn.Parameter(torch.randn(1)) for _ in range(5)]
        self.model.state_dict.return_value = {}
        self.criterion = MagicMock()
        self.model.epochs_in_current_stage = 0
        self.model.unfreeze_state = {
            'stage_history' : [],
            'current_stage': 0,
            'total_stages': 5
        }
        self.config = {
            'early_stop_patience': 3,
            'min_improvement': 0.001,
            'stage_lrs': [0.001, 0.0005, 0.0001, 0.00005 ,0.00001],
            'weight_decay': 1e-5,
            'head_epochs': 2,
            'stage_epochs': 5
        }
        print("\n--- Starting New Test ---")
    
    def tearDown(self):
        print("---Test Completed---\n")

        return super().tearDown()
    def adaptive_unfreeze_side_effect(self, patience_reached=False):
    
        new_stage = False
        current_stage = self.model.unfreeze_state['current_stage']

        if current_stage == 0 and self.model.epochs_in_current_stage > self.config['head_epochs']:
            new_stage = True
        elif current_stage > 0 and current_stage < self.model.unfreeze_state['total_stages'] - 1:
            if self.model.epochs_in_current_stage > self.config['stage_epochs'] or patience_reached:
                new_stage = True
        
        if new_stage:
            self.model.unfreeze_state['current_stage'] += 1
            self.model.epochs_in_current_stage = 0
            self.model.unfreeze_state['stage_history'].append((self.model.unfreeze_state['current_stage'], 'epoch' if not patience_reached else 'performance'))
        else:
            self.model.epochs_in_current_stage += 1

        return new_stage

    @patch('simple_ft.simple_ft.train_epoch')
    @patch('simple_ft.simple_ft.validate')
    @patch('torch.optim.AdamW')
    # Simulate early stopping. 1 stage change, 2nd needs to break not change.
    def test_early_stopping(self, mock_AdamW, mock_validate, mock_train_epoch):
        print("Running test early stop")
        self.model.epochs_in_current_stage = 0
        num_epochs = 30
        mock_train_epoch.return_value = (0.5, 80.0)
        val_losses = [0.4, 0.39, 0.38] + [0.37 + i*0.01 for i in range(27)]
        val_accs =  [82.0, 84.5, 83.0]+ [83.0 - i*0.5 for i in range(27)]
        
        mock_validate.side_effect = list(zip(val_losses, val_accs))
        mock_callback = MockCallback()

        mock_AdamW.side_effect = DynamicMockOptimizer
        
        self.model.adaptive_unfreeze.side_effect = self.adaptive_unfreeze_side_effect

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                     num_epochs, self.config, callback=mock_callback)
        exp_epochs = 2 + 3 + 3
        self.assertEqual(len(results['val_accs']), exp_epochs, f"More epochs than expected {exp_epochs}")
        self.assertEqual(self.model.unfreeze_state['current_stage'], 2, "Training should stop at the first stage, as no improvements")
        self.assertEqual(results['best_val_acc'], 84.5, "Best accuracy should be 84.5")
        
        # Check stage transitions
        expected_stages = [(1, 'epoch'), (2, 'performance')]
        self.assertEqual(self.model.unfreeze_state['stage_history'], expected_stages)
        # check if lr has chnaged 
        self.assertEqual(mock_callback.calls[0]['lr'], self.config['stage_lrs'][0], f"Learning rate of head is not correct in epoch 1") 
        self.assertEqual(mock_callback.calls[1]['lr'], self.config['stage_lrs'][0], f"Learning rate of stage 0 is not correct")
        self.assertEqual(mock_callback.calls[2]['lr'], self.config['stage_lrs'][1], f"Learning rate should chnage at begining of epoch 3")
        self.assertEqual(mock_callback.calls[-1]['lr'], self.config['stage_lrs'][2],f"The learning rate doesnt match stage 2 learning rate")
                

    @patch('simple_ft.simple_ft.train_epoch')
    @patch('simple_ft.simple_ft.validate')
    @patch('torch.optim.AdamW')
    def test_continuous_improvement(self, mock_AdamW, mock_validate, mock_train_epoch):
        print("Test continuous improvements")
        self.model.epochs_in_current_stage = 0
        num_epochs = 30
        mock_train_epoch.return_value = (0.5, 80.0)
        val_losses = [0.4 - i*0.01 for i in range(num_epochs)]
        val_accs = [70.0 + i*0.5 for i in range(num_epochs)]
        mock_validate.side_effect = list(zip(val_losses, val_accs))
        mock_callback = MockCallback()

        mock_AdamW.side_effect = DynamicMockOptimizer
        
        self.model.adaptive_unfreeze.side_effect = self.adaptive_unfreeze_side_effect

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                    num_epochs, self.config, callback=mock_callback)
        
        self.assertEqual(len(results['val_accs']), num_epochs, "Training should complete all epochs")
        self.assertEqual(self.model.unfreeze_state['current_stage'], 4, "All stages should be unfrozen")
        self.assertEqual(results['best_val_acc'], val_accs[-1], "Best accuracy should be the last one")
        self.assertIn('best_model_state', results, "Best model state not saved in results")
        self.assertIsNotNone(results['best_model_state'], "Best model state is None")
        
        # Check stage transitions
        expected_stages = [(1, 'epoch'), (2, 'epoch'), (3, 'epoch'), (4, 'epoch')]
        self.assertEqual(self.model.unfreeze_state['stage_history'], expected_stages)

        
        self.assertEqual(mock_callback.calls[0]['lr'], self.config['stage_lrs'][0], f"Learning rate of head is not correct")
        self.assertEqual(mock_callback.calls[3]['lr'], self.config['stage_lrs'][1], f"Learning rate of")
        #self.assertEqual(mock_callback.calls('lr'), self.config['stage_lrs'][2], f"Learning rate of ")
        #self.assertEqual(mock_callback.calls('lr'), self.config['stage_lrs'][3], f"Learning r")
        #self.assertEqual(mock_callback.calla('lr'), self.config['stage_lrs'][4], f"Learning rate")

       
    
    @patch('simple_ft.simple_ft.train_epoch')
    @patch('simple_ft.simple_ft.validate')
    @patch('torch.optim.AdamW')
        #Initialise at 3rd stage to go to 4th 
    def test_early_stop_final_stage(self, mock_AdamW, mock_validate, mock_train_epoch):
        print("Test early stopping final stage")
        self.model.epochs_in_current_stage = 0
        num_epochs = 30
        self.model.unfreeze_state = {'stage_history': [(1, 'epoch'), (2, 'epoch'), (3, 'epoch')], 'current_stage': 3, 
                                     'total_stages': 5}
        mock_train_epoch.return_value = (0.6, 80.0)

        val_losses = [0.4, 0.41, 0.42]  + [0.41 + i*0.01 for i in range(num_epochs)]
        val_accs = [ 82.0, 82.5, 83.0]  + [83.0 - i*0.5 for i in range(num_epochs)]                           

        mock_validate.side_effect = list(zip(val_losses, val_accs))
        mock_AdamW.side_effect = DynamicMockOptimizer
        mock_callback = MockCallback()

        self.model.adaptive_unfreeze.side_effect = self.adaptive_unfreeze_side_effect

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                     num_epochs, self.config, callback=mock_callback)

        self.assertEqual(self.model.unfreeze_state['current_stage'], 4, "Model did not reach stage 4")

        for i, call in enumerate(mock_callback.calls):
            if i == 2:
                self.assertEqual(call['lr'], self.config['stage_lrs'][3], f"Learning rate of stage 3 is not correct {call['lr']}")
            elif i > 6: 
                self.assertEqual(call['lr'], self.config['stage_lrs'][4], f"Learning rate shouldnt change")
        
    @patch('simple_ft.simple_ft.train_epoch')
    @patch('simple_ft.simple_ft.validate')
    @patch('torch.optim.AdamW')
    def test_stagnation_n_improv(self, mock_AdamW, mock_validate, mock_train_epoch):
        print("Test stagnation n improvement")
        self.model.epochs_in_current_stage = 0
        num_epochs = 30
        mock_train_epoch.return_value = (0.5, 80.0)
        val_losses = [0.4, 0.39, 0.37] + [0.37] * 3 + [0.4 - i*0.01 for i in range(24)]
        val_accs =  [82.0, 82.5, 83.0] + [83.0] * 3 + [84.0 + i*0.05 for i in range(24)]
                             #           #                                                                       #
        mock_validate.side_effect = list(zip(val_losses, val_accs))
        mock_callback = MockCallback()

        mock_AdamW.side_effect = DynamicMockOptimizer
        
        self.model.adaptive_unfreeze.side_effect = self.adaptive_unfreeze_side_effect

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                     num_epochs, self.config, callback=mock_callback)
        
        self.assertEqual(self.model.unfreeze_state['current_stage'], 4, f"All stages should be unfrozen")
        self.assertEqual(mock_callback.calls[-1]['lr'], self.config['stage_lrs'][4], f"End learning rate not correct")
        
        ex_stage_history = [(1, 'epoch'), (2, 'performance'), (3, 'epoch'), (4, 'epoch')]
        self.assertEqual(self.model.unfreeze_state['stage_history'], ex_stage_history)

if __name__ == '__main__':
    unittest.main()