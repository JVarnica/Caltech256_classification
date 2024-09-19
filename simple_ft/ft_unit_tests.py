import sys
from unittest.mock import MagicMock

"""sys.modules['linear_probe'] = MagicMock()
sys.modules['linear_probe.model_eval'] = MagicMock()
sys.modules['linear_probe.model_eval'].get_dali_loader = MagicMock()

sys.modules['models'] = MagicMock()
sys.modules['models.bs_model_wrapper'] = MagicMock()
sys.modules['models.bs_model_wrapper'].BaseTimmWrapper = MagicMock()"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from simple_ft import train_epoch, validate, handle_new_stage, save_results, get_exp_config
import os
import tempfile

class TestSimpleFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setUp(self):
        self.model = nn.Linear(3 * 224 * 224, 10)
        self.model.to(self.device)
        self.mock_data = torch.randn(100, 3, 224, 224)
        self.mock_labels = torch.radint(0, 10, (100,))
        self.dataset = TensorDataset(self.mock_data, self.mock_labels)
        self.dataloader = DataLoader(self.dataset, batch_size=32)
        self.criterion = nn.CrossEntropyLoss()


    def test_train_epoch(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay= 0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience= 1)
        stage_checkpoint = self.model.state_dict().copy()
        stage_best_val_acc = 0.5
        stage_results = []

        new_opti, new_sch, new_stage_best_acc, new_stage_checkpoint = handle_new_stage(self.model, optimizer, scheduler, stage_best_val_acc, stage_checkpoint, stage_results, 1, 0.001, 0.01, 5)

        self.assertIsInstance(new_opti, torch.optim.AdamW)
        self.assertIsInstance(new_sch, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(new_stage_best_acc, 0)
        self.assertIsNone(new_stage_checkpoint)
       
    
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

    def test_get_exp_config(self):
        config = get_exp_config('caltech256')
        self.assertIsInstance(config, dict)
        self.assertIn('model_list', config)
        self.assertIn('train_dir', config)
        self.assertIn('val_dir', config)
        self.assertIn('results_dir', config)
        self.assertIn('batch_size', config)
        self.assertIn('num_classes', config)


if __name__ == '__main__':
    print("Startign unittest main")
    unittest.main()

    