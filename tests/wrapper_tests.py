import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from models.bs_model_wrapper import BaseTimmWrapper

class TestTimmWrapper(unittest.TestCase):
    def setUp(self):
        self.models_list = [
            'resnet18.a1_in1k',
            'vit_base_patch16_224.orig_in21k_ft_in1k',
            'regnety_040.pycls_in1k',
            'pvt_v2_b3.in1k'
        ]
        self.num_classes = 10 
    
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
    
    def test_adaptive_unfreeze(self):
        for model_name in self.models_list:
            with self.subTest(model=model_name):

                model = BaseTimmWrapper(model_name, self.num_classes, freeze_mode='gradual', unfreeze_epochs=[1, 3, 5, 7])
                # Initial state test need all groups frozen
                self.assertEqual(model.unfreeze_state['current_stage'], 0, f"Current stage must be zero, but at {model.unfreeze_state['current_stage']}")
                self.assertFalse(any(p.requires_grad for group in model.param_groups for p in group['params']), 
                                 f"All param groups should be frozen at stage0") 

                #Epoch based unfreezing
                new_stage = model.adaptive_unfreeze(1, 0.5, False)
                self.assertTrue(new_stage, f"Stage should change at epoch 1 for {model_name}")
                self.assertEqual(model.unfreeze_state['current_stage'], 1, f"Current stage should be 1 after unfreeze")
                self.assertTrue(all(p.requires_grad for p in model.param_groups[0]['params']),
                                f"First param group should be unfrozen now for {model_name}")
                self.assertFalse(any(p.requires_grad for group in model.param_groups[1:] for p in group['params']),
                                 f"Later param groups should remain frozen after first unfreeze {model_name}")
                
                #No unfreezing should occur rndm epoch
                new_stage = model.adaptive_unfreeze(2, 0.6, False)
                self.assertFalse(new_stage, f"There should have been no change")
                self.assertEqual(model.unfreeze_state['current_stage'], 1, f"Current stage should remain 1 {model_name}")

                # patience unfreeze
                new_stage = model.adaptive_unfreeze(4, 0.7, True)
                self.assertTrue(new_stage, f"Patience reached stage shoulda chnaged!! {model_name}")
                self.assertEqual(model.unfreeze_state['current_stage'], 2, F"Current stage should be two after two unfreezes {model_name}")

                model.adaptive_unfreeze(6, 0.8, False)
                self.assertEqual(model.unfreeze_state['best_performance'], 0.9, f"Best performance should have updated!!")

                #Test if stage history is updated
                expected_hist = [(1, 'epoch'), (2, 'performance')]
                self.assertEqual(model.unfreeze_state['stage_history'], expected_hist,
                                 f"Stage History should match expected history")
                
                 # Test no more unfreezing after all stages are unfrozen
                for _ in range(len(model.param_groups) - model.unfreeze_state['current_stage']):
                    model.adaptive_unfreeze(8, 0.6, True)
                final_stage = model.unfreeze_state['current_stage']
                new_stage = model.adaptive_unfreeze(9, 0.9, True)
                self.assertFalse(new_stage, f"No new stage should occur after all unfreezing for {model_name}")
                self.assertEqual(model.unfreeze_state['current_stage'], final_stage, 
                                 f"Current stage should equal final stage, it can chnage no more stages{model_name}")

    def test_full_finetune(self):
        model = BaseTimmWrapper(self.models_list[0], self.num_classes, freeze_mode='gradual')
        result = model.full_finetune()
        self.assertTrue(result)
        for param in model.parameters():
            self.assertTrue(param.requires_grad)
        self.assertEqual(model.unfreeze_state['current_stage'], model.unfreeze_state['total_stages'])


if __name__ == '__main__':
    unittest.main()
