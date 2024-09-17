import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from models.bs_model_wrapper import BaseTimmWrapper
from simple_ft import train_epoch, validate, handle_new_stage, save_results, get_exp_config, 
import os
import shutil
import tempfile

class TestSimpleFT(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.model = BaseTimmWrapper("resnet18.a1_in1k", num_classes=10, freeze_mode='gradual', unfreeze_epochs=[1, 2])
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.mock_data = torch.randn(100, 3, 224, 224)
        self.mock_labels = torch.radint(0, 10, (100,))
        self.dataset = TensorDataset(self.mock_data, self.mock_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=32)

    def test_train_epoch(self):
        optimizer = torch.optim.AdamW(self.model.get_trainable_params())
        loss, acc = train_epoch(self.model, self.dataloader, optimizer, self.criterion, self.device)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertLessEqual(acc, 100)
    
    def test_validate(self):
        loss, acc = validate(self.model, self.dataloader, self.criterion, self.device)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertLessEqual(acc, 100)

    def test_handle_new_stages(self):
        optimizer = torch.optim.AdamW(self.model.get_trainable_params(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience= 1)
        stage_best_val_acc = 0.5
        stage_checkpoint = self.model.state_dict().copy()
        stage_results = []

        new_opti, new_sch, new_stage_best_acc, new_stage_checkpoint = handle_new_stage(self.model, optimizer, scheduler, stage_best_val_acc, stage_checkpoint, stage_results, 1, 1e-4, 1e-5, 5)

        self.assertIsInstance(new_opti, torch.optim.AdamW)
        self.assertIsInstance(new_sch, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(new_stage_best_acc, 0)
        self.assertIsNone(new_stage_checkpoint)

    def test_get_exp_config(self):
        config = get_exp_config('caltech256')
        self.assertIsInstance(config, dict)
        self.assertIn('model_list', config)
        self.assertIn('train_dir', config)
        self.assertIn('val_dir', config)\
    
    def test_save_results(self):
        results = {
            'train_losses': [0.5, 0.4],
            'train_accs': [80, 85],
            'val_losses': [0.6, 0.5],
            'val_accs': [75, 80],
            'epoch_times': [10, 11]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_results(results, tmpdir, 'test_model')
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'test_model_metrics.csv')))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'test_model_learning_curves.png')))

    def test_adaptive_unfreeze(self):
        initial_frozen = sum(1 for p in self.model.get_trainable_params())
        self.model.adaptive_unfreeze(1, 10, 0.5)
        after_unfreeze = sum(1 for p in self.model.get_trainable_params())
        self.assertLess(after_unfreeze, initial_frozen)


if __name__ == '__main__':
    unittest.main()