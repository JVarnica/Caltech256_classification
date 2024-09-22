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
                for param in model_full.base_model.parameters():
                    self.assertFalse(param.requires_grad)
                model_none = BaseTimmWrapper(model_name, self.num_classes, freeze_mode='none')
                for param in model_none.base_model.parameters():
                    self.assertFalse(param.requires_grad)
                model_gradual = BaseTimmWrapper(model_name, self.num_classes, freeze_mode='gradual')
                for param in model_gradual.base_model.parameters():
                    self.assertFalse(param.requires_grad)
                