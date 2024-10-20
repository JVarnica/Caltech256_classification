import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from simple_ft.simple_ft import train_and_evaluate
from models.bs_model_wrapper import BaseTimmWrapper


class MockCallback:
    def __init__(self):
        self.calls = []
    
    def __call__(self, epoch, model, optimizer ,train_loss, train_acc, val_loss, val_acc, lr):
        self.calls.append({
            'epoch': epoch,
            'model_stage': model.unfreeze_state['current_stage'],
            'optimizer': optimizer,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        })

class MockModel(BaseTimmWrapper):
    def __init__(self):
        super().__init__('vit_base_patch16', num_classes=10, freeze_mode='gradual') 
        self.model_name = "TestModel"
        self.freeze_mode = 'gradual'
        self.head_epochs = 2
        self.stage_epochs = 4
        self.epochs_in_current_stage = 0
        self.unfreeze_state = {
            'stage_history' : [],
            'current_stage': 0,
            'total_stages': 5
        }
        self.dummylayer = nn.Linear(10, 10)

    def forward(self, x):
        return self.dummylayer(x)
    
    def get_trainable_params(self):
        return self.parameters()
    
    def state_dict(self):
        return super().state_dict()
    #Copy pasted so exactly same
    def adaptive_unfreeze(self, patience_reached=False):
        if self.freeze_mode != 'gradual':
            return False

        current_stage = self.unfreeze_state['current_stage']
        total_stages = self.unfreeze_state['total_stages']
        stage_history = self.unfreeze_state['stage_history']
        new_stage = False

        if current_stage == 0 and self.epochs_in_current_stage >= self.head_epochs - 1: # Just for head 
            new_stage = True
        elif current_stage > 0 and current_stage < total_stages - 1: # Dont wanna unfreeze patch embeddings
            if self.epochs_in_current_stage >= self.stage_epochs -1  or patience_reached:
                new_stage = True
        elif current_stage == total_stages - 1 and patience_reached:
            return 'final_stage_patience'
        
        if patience_reached:
            if len(stage_history) >=1 and stage_history[-1][1] == 'performance':
                return 'early_stop'
        
        if new_stage:
            current_stage += 1
            self.epochs_in_current_stage = 0
            stage_history.append((current_stage, 'epoch' if not patience_reached else 'performance'))
            for param in self.param_groups[current_stage - 1]['params']:
                if isinstance(param, list):
                    for p in param:
                        if hasattr(p, 'requires_grad'):
                            p.requires_grad = True
                elif hasattr(param, 'requires_grad'):
                    param.requires_grad = True
        else:
            self.epochs_in_current_stage += 1
        #Make sure update
        self.unfreeze_state['current_stage'] = current_stage
        self.unfreeze_state['stage_history'] = stage_history

        return new_stage

class DynamicMockOptimizer:
    def __init__(self, *args, **kwargs):
        self.param_groups = [{'lr': kwargs['lr']}]

class TestTrain_Evaluate(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MockModel()
        self.criterion = MagicMock()
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
        

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                     num_epochs, self.config, callback=mock_callback)
        exp_epochs = 2 + 3 + 3
        self.assertEqual(len(results['val_accs']), exp_epochs, f"More epochs than expected {exp_epochs}")
        self.assertEqual(self.model.unfreeze_state['current_stage'], 2, "Training should stop at the second stage, as no improvements")
        self.assertEqual(results['best_val_acc'], 84.5, "Best accuracy should be 84.5")
        
        # Check stage transitions
        expected_stages = [(1, 'epoch'), (2, 'performance')]
        self.assertEqual(self.model.unfreeze_state['stage_history'], expected_stages)
        # check if lr has chnaged 
        self.assertEqual(mock_callback.calls[0]['lr'], self.config['stage_lrs'][0], f"Learning rate of head is not correct in epoch 1") 
        self.assertEqual(mock_callback.calls[1]['lr'], self.config['stage_lrs'][0], f"Learning rate of stage 0 is not correct")
        self.assertEqual(mock_callback.calls[2]['lr'], self.config['stage_lrs'][1], f"Learning rate should chnage at begining of epoch 3")
        self.assertEqual(mock_callback.calls[-1]['lr'], self.config['stage_lrs'][2],f"The learning rate doesnt match stage 2 learning rate")

        print(f"Final Stage: {self.model.unfreeze_state['current_stage']}")
                

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
        self.model.unfreeze_state = {'stage_history': [(1, 'epoch'), (2, 'epoch'), (3, 'epoch'), (4, 'epoch')], 
                                     'current_stage': 4, 
                                     'total_stages': 5}
        mock_train_epoch.return_value = (0.6, 80.0)

        val_losses = [0.41 + i*0.01 for i in range(num_epochs)]
        val_accs = [ 83.0 - i*0.5 for i in range(num_epochs)]                           
        mock_validate.side_effect = list(zip(val_losses, val_accs))
        mock_callback = MockCallback()

        mock_AdamW.return_value = DynamicMockOptimizer

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                     num_epochs, self.config, callback=mock_callback)
        
        exp_epochs = 4
        self.assertEqual(len(results['val_accs']), exp_epochs, f"Expected {exp_epochs} epochs, got {len(results['val_accs'])}")

        self.assertEqual(self.model.unfreeze_state['current_stage'], 4, "Model should early stop")
        self.assertEqual(len(self.model.unfreeze_state['stage_history']), 4, "No new stage transitions should have occurred")
        # No learning rate assertions cannot make it start at stage 3 currently and no hugely important just need to know it is at the right stage.
        
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

        results = train_and_evaluate(self.model, MagicMock(), MagicMock(), self.criterion, self.device,
                                     num_epochs, self.config, callback=mock_callback)
        
        self.assertEqual(self.model.unfreeze_state['current_stage'], 4, f"All stages should be unfrozen")
        self.assertEqual(mock_callback.calls[-1]['lr'], self.config['stage_lrs'][4], f"End learning rate not correct")
        
        ex_stage_history = [(1, 'epoch'), (2, 'performance'), (3, 'epoch'), (4, 'epoch')]
        self.assertEqual(self.model.unfreeze_state['stage_history'], ex_stage_history)

if __name__ == '__main__':
    unittest.main()