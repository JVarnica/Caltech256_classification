import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock
import torch
from simple_ft.adaptive_ft_main import train_and_evaluate
from models.bs_model_wrapper import BaseTimmWrapper

class MockCallback:
   def __init__(self):
       self.calls = []
   
   def __call__(self, epoch, model, optimizer, train_loss, train_acc, val_loss, val_acc, lr):
       self.calls.append({
           'epoch': epoch,
           'stage': model.current_stage if hasattr(model, 'current_stage') else 0,
           'optimizer': optimizer,
           'train_loss': train_loss,
           'train_acc': train_acc,
           'val_loss': val_loss,
           'val_acc': val_acc,
           'lr': lr
       })

class DynamicMockOptimizer:
   def __init__(self, *args, **kwargs):
       self.param_groups = [{'lr': kwargs.get('lr', 0.001)}]

class TestTrainEvaluate(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = MagicMock()
        self.config = {
           'early_stop_patience': 3,
           'plateau_patience': 5,
           'min_improvement': 0.001,
           'base_lr': 0.001,
           'weight_decay': 1e-5,
           'head_epochs': 2,
           'stage_epochs': 4,
           'max_adapt': 5
       }
        print("\n--- Starting New Test ---")

    @patch('simple_ft.adaptive_ft_main_2.get_dali_loader')
    @patch('simple_ft.adaptive_ft_main.train_epoch')
    @patch('simple_ft.adaptive_ft_main.validate')
    @patch('torch.optim.AdamW')
    def test_continuous_improvement(self, mock_AdamW, mock_validate, mock_train_epoch, mock_get_dali_loader):
       """Test both strategies with continuous improvement to check stage/LR changes"""
       num_epochs = 30
       models_to_test = [
           ('resnet50.a1_in1k', 'gradual'),
           ('vit_base_patch16_224.orig_in21k_ft_in1k', 'discriminative'),
           ('pvt_v2_b3.in1k', 'gradual'),
           ('regnety_040.pycls_in1k', 'full')
       ]

       for model_name, strategy in models_to_test:
           with self.subTest(model=model_name, strategy=strategy):
               print(f"\nTesting {model_name} with {strategy} strategy")
               model = BaseTimmWrapper(model_name, num_classes=10, 
                                     ft_strategy=strategy,
                                     head_epochs=self.config['head_epochs'],
                                     stage_epochs=self.config['stage_epochs'])
               
               mock_get_dali_loader.return_value = MagicMock()
               mock_train_epoch.return_value = (0.5, 80.0)
               val_losses = [0.4 - i*0.01 for i in range(num_epochs)]
               val_accs = [70.0 + i*0.5 for i in range(num_epochs)]
               mock_validate.side_effect = list(zip(val_losses, val_accs))

               mock_callback = MockCallback()
               mock_AdamW.side_effect = DynamicMockOptimizer


               results = train_and_evaluate(model, MagicMock(), MagicMock(), 
                                         self.criterion, self.device,
                                         num_epochs, self.config, callback=mock_callback)
               if strategy == 'full':
                    # Verify all params trainable
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    self.assertEqual(trainable_params, total_params, 
                                    f"All parameters should be trainable for {model_name}")
                    
                    # Verify learning rate decreases (cosine schedule)
                    lrs = [call['lr'] for call in mock_callback.calls]
                    self.assertGreater(lrs[0], lrs[-1], 
                                    f"Learning rate should decrease with cosine schedule for {model_name}")
                    self.assertEqual(len(set(lrs)), len(lrs), 
                                    f"Learning rate should change every epoch for {model_name}")
               elif strategy == 'gradual':
                   # Verify stage progression
                   stage_changes = [i for i, call in enumerate(mock_callback.calls[1:], 1) 
                                  if call['stage'] != mock_callback.calls[i-1]['stage']]
                   self.assertEqual(len(stage_changes), 4, f"Should have 4 stage changes for {model_name}")

               elif strategy == 'discriminative':
                   # All params should be trainable from start
                   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                   total_params = sum(p.numel() for p in model.parameters())
                   self.assertEqual(trainable_params, total_params, 
                                  f"All parameters should be trainable for {model_name}")

               self.assertEqual(len(results['val_accs']), num_epochs, 
                              f"Should complete all epochs for {model_name}")
               self.assertGreater(results['best_val_acc'], val_accs[0], 
                                f"Should show improvement for {model_name}")
               
    @patch('simple_ft.adaptive_ft_main_2.get_dali_loader')
    @patch('simple_ft.adaptive_ft_main.train_epoch')
    @patch('simple_ft.adaptive_ft_main.validate')
    @patch('torch.optim.AdamW')
    def test_plateau_and_adaptation(self, mock_AdamW, mock_validate, mock_train_epoch, mock_get_dali_loader):
        """Test both strategies with plateaus to verify adaptations and early stopping"""
        num_epochs = 50
        models_to_test = [
           ('resnet50.a1_in1k', 'full'),
           ('vit_base_patch16_224.orig_in21k_ft_in1k', 'discriminative'),
           ('regnety_040.pycls_in1k', 'full')
       ]

        for model_name, strategy in models_to_test:
            with self.subTest(model=model_name, strategy=strategy):
                print(f"\nTesting {model_name} with {strategy} strategy - plateau behavior")
                model = BaseTimmWrapper(model_name, num_classes=10, 
                                     ft_strategy=strategy,
                                     head_epochs=self.config['head_epochs'],
                                     stage_epochs=self.config['stage_epochs'])

               # Pattern: improve -> plateau -> improve -> plateau -> final plateau
                val_losses = []
                val_accs = []
               
               # Initial improvement
                for i in range(5):
                   val_losses.append(0.4 - i*0.01)
                   val_accs.append(70 + i)
               
               # First plateau - trigger adaptation
                for _ in range(4):
                   val_losses.append(val_losses[-1])
                   val_accs.append(val_accs[-1])
               
               # Second improvement phase
                for i in range(5):
                   val_losses.append(val_losses[-1] - 0.01)
                   val_accs.append(val_accs[-1] + 1)
               
               # Second plateau - trigger adaptation
                for _ in range(4):
                   val_losses.append(val_losses[-1])
                   val_accs.append(val_accs[-1])
               
               # Final plateau to trigger early stopping
                for _ in range(6):
                   val_losses.append(val_losses[-1])
                   val_accs.append(val_accs[-1])
                
                mock_get_dali_loader.return_value = MagicMock()
                mock_train_epoch.return_value = (0.5, 80.0)
                mock_validate.side_effect = list(zip(val_losses, val_accs))
                mock_callback = MockCallback()
                mock_AdamW.side_effect = DynamicMockOptimizer

                results = train_and_evaluate(model, MagicMock(), MagicMock(), 
                                         self.criterion, self.device,
                                         num_epochs, self.config, callback=mock_callback)

                # Verify adaptations and early stopping
                if strategy == 'gradual':
                   stage_changes = [i for i, call in enumerate(mock_callback.calls[1:], 1) 
                                  if call['stage'] != mock_callback.calls[i-1]['stage']]
                   self.assertEqual(len(stage_changes), 2, 
                                  f"Should have 2 forced stage changes for {model_name}")
                   
                elif strategy == 'full':
                    lrs = [call['lr'] for call in mock_callback.calls]
                    self.assertGreater(lrs[0], lrs[-1], 
                      f"Learning rate should decrease with cosine schedule for {model_name}")            
            
                elif strategy == 'discriminative':
                   lr_changes = [i for i in range(1, len(mock_callback.calls)) 
                               if mock_callback.calls[i]['lr'] != mock_callback.calls[i-1]['lr']]
                   self.assertEqual(len(lr_changes), 2, 
                                  f"Should have 2 LR reductions for {model_name}")

               # Verify plateau stopping
                self.assertEqual(len(results['val_accs']), 24, 
                              f"Should stop at plateau for {model_name}")

if __name__ == '__main__':
   unittest.main()