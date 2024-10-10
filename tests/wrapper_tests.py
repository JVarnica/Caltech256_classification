import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock, patch
from models.bs_model_wrapper import BaseTimmWrapper
from simple_ft.simple_ft import train_and_evaluate

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TestTimmWrapper(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_list = [
            'resnet50.a1_in1k',
            'vit_base_patch16_224.orig_in21k_ft_in1k',
            'regnety_040.pycls_in1k',
            'pvt_v2_b3.in1k'
        ]
        self.num_classes = 10
        self.config = {
            'early_stop_patience': 3,
            'min_improvement': 0.001,
            'stage_lrs': [0.001, 0.0005, 0.0001, 0.00005, 0.00001],
            'weight_decay': 1e-5,
            'head_epochs': 2,
            'stage_epochs': 4,
            'num_epochs': 30
        }
    
    def wrapper_initialization(self):
        for model_name in self.models_list:
            with self.subTest(model=model_name):
                model = BaseTimmWrapper(model_name, self.num_classes, freeze_mode='gradual')
                self.assertIsInstance(model, BaseTimmWrapper)
                self.assertEqual(model.freeze_mode, 'gradual')
                self.assertIn('total_stages', model.unfreeze_state)
                if 'resnet' in model_name:
                    self.assertEqual(model.base_model.fc.out_features, self.num_classes)
                elif 'vit'  or 'pvt_v2' in model_name:
                    self.assertEqual(model.base_model.head.out_features, self.num_classes)
                elif 'regnety' in model_name:
                    self.assertEqual(model.base_model.head.fc.out_features, self.num_classes)
                else:
                    self.fail(f"Test not implemented for {model_name}")
    
    def test_freeze_mode(self):
        for model_name in self.models_list:
            with self.subTest(model=model_name):

                model_full = BaseTimmWrapper(model_name, self.num_classes, freeze_mode='full')
                for name, param in model_full.base_model.named_parameters():
                    if not any(head_name in name for head_name in ['fc', 'head']):
                        self.assertFalse(param.requires_grad, f"Parameter: {name} should be frozen in {model_name}")

                model_none = BaseTimmWrapper(model_name, self.num_classes, freeze_mode='none')
                for name, param in model_none.base_model.named_parameters():
                    if not any(head_name in name for head_name in ['fc', 'head']):
                        self.assertTrue(param.requires_grad, f"In none mode parameter {name} should be trainable for {model_name}")

                model_gradual = BaseTimmWrapper(model_name, self.num_classes, freeze_mode='gradual')
                for name, param in model_gradual.base_model.named_parameters():
                    if not any(head_name in name for head_name in ['fc', 'head']):
                        self.assertFalse(param.requires_grad, f" In gradual at initial stage {name} should be frozen for {model_name}")

        with self.assertRaises(ValueError):
            BaseTimmWrapper(self.models_list[0], self.num_classes, freeze_mode='True') #test invalid input
  
    def test_full_finetune(self):
        model = BaseTimmWrapper(self.models_list[0], self.num_classes, freeze_mode='gradual')
        result = model.full_finetune()
        self.assertTrue(result)
        for param in model.parameters():
            self.assertTrue(param.requires_grad)
        self.assertEqual(model.unfreeze_state['current_stage'], model.unfreeze_state['total_stages'])


    @patch('simple_ft.simple_ft.train_epoch')
    @patch('simple_ft.simple_ft.validate')
    def test_multiple_models_training(self, mock_validate, mock_train_epoch):
        for model_name in self.models_list:
            with self.subTest(model=model_name):
                print(f"\nTesting {model_name}")
                
                model = BaseTimmWrapper(model_name, self.num_classes, freeze_mode='gradual',
                                        head_epochs=self.config['head_epochs'],
                                        stage_epochs=self.config['stage_epochs'])
                
                criterion = nn.CrossEntropyLoss()

                mock_train_epoch.return_value = (0.5, 80.0)
                val_losses = [0.4 - i*0.01 for i in range(self.config['num_epochs'])]
                val_accs = [70.0 + i+0.5 for i in range(self.config['num_epochs'])]
                mock_validate.side_effect = list(zip(val_losses, val_accs))


                initial_trainable_params = count_trainable_params(model)
                print(f"Initial trainable parameters: {initial_trainable_params}")

                learning_rates = []
                stages = []
                trainable_params = []

                def callback(epoch, model, optimizer, train_loss, train_acc, val_loss, val_acc, current_lr):
                    learning_rates.append(current_lr)
                    stages.append(model.unfreeze_state['current_stage'])
                    trainable_params.append(count_trainable_params(model))
                    print(f"Epoch {epoch}, Stage {model.unfreeze_state['current_stage']}, "
                      f"Trainable params: {trainable_params[-1]}, LR: {current_lr}")


                results = train_and_evaluate(model, MagicMock(), MagicMock(), criterion, self.device,
                                             self.config['num_epochs'], self.config, callback=callback)

                final_trainable_params = count_trainable_params(model)
                print(f"Final trainable parameters: {final_trainable_params}")

                self.assertGreater(final_trainable_params, initial_trainable_params,
                                   f"Trainable parameters should increase for {model_name}")

                # Check if stages progressed
                self.assertGreater(max(stages), 0, f"Model should progress through stages for {model_name}")

                # Check if learning rate changed
                unique_lrs = set(learning_rates)
                self.assertGreater(len(unique_lrs), 1, f"Learning rate should change for {model_name}")

                print(f"Stages progressed: {model.unfreeze_state['stage_history']}")
                print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")

                # Print parameter counts for each stage
                for stage, params in enumerate(trainable_params):
                    if stage == 0 or params != trainable_params[stage-1]:
                        print(f"Stage {stage}: {params} trainable parameters")

if __name__ == '__main__':
    unittest.main()
