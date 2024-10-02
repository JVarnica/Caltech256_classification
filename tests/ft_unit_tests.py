import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from simple_ft.simple_ft import train_epoch, validate, save_results, get_exp_config

#Mock configs module import
sys.modules['configs'] = MagicMock()
sys.modules['configs.simple_ft_cal256_config'] = MagicMock()

class DaliLoaderMock:
    def __init__(self, batch_size, num_batches, num_classes):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_classes = num_classes
    
    def __iter__(self):
        for _ in range(self.num_batches):
            yield [{'image': torch.randn(self.batch_size, 3, 224, 224),
                    'label': torch.randint(0, self.num_classes, (self.batch_size,))}]
    
    def __len__(self):
        return self.num_batches
    
class MockModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)
        

class TestSimpleFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.mock_config = {
            'model_list': [{'model_name': 'test_model', 'freeze_mode': 'gradual'}],
            'train_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/train',
            'val_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/val',
            'results_dir': '/content/drive/MyDrive/caltech_proj/caltech_proj/simple_ft/sft_cal_results',
            'num_classes': 10,
            'batch_size': 32,
            'num_threads': 8,
            'num_epochs': 10,
            'learning_rate': 0.01,
            'weight_decay': 0.001,
            'early_stop_patience': 5,
            'scheduler_patience': 3,
            'min_improvement': 0.001
        }

    def setUp(self):
        self.num_classes = self.mock_config['num_classes']
        self.model = MockModel(self.num_classes).to(self.device)
        self.batch_size = self.mock_config['batch_size']
        self.num_batches = 5
        self.train_loader = DaliLoaderMock(self.batch_size, self.num_batches, self.num_classes)
        self.val_loader = DaliLoaderMock(self.batch_size, self.num_batches, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def test_train_epoch(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        loss, acc = train_epoch(self.model, self.train_loader, optimizer, self.criterion, self.device)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertLessEqual(acc, 100)
    
    def test_validate(self):
        loss, acc = validate(self.model, self.val_loader, self.criterion, self.device)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertLessEqual(acc, 100)
       
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

    @patch('importlib.import_module')
    def test_get_exp_config(self, mock_import):
        mock_module = MagicMock()
        mock_module.SIMPLE_FT_CALTECH256_CONFIG = self.mock_config
        mock_import.return_value = mock_module

        config = get_exp_config('caltech256')

        self.assertIsInstance(config, dict)
        self.assertEqual(config, self.mock_config)
       

if __name__ == '__main__':
    print("Startign unittest main")
    unittest.main()

    