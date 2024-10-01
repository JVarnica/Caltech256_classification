SIMPLE_MOCK_DATA_CONFIG = {
    'model_list' : [
           {
            'model_name': 'resnet50.a1_in1k', 
            'freeze_mode': 'gradual',
          'unfreeze_epochs': [0, 1]
        },
        {
            'model_name': 'regnety_040.pycls_in1k', 
            'freeze_mode': 'gradual',
            'unfreeze_epochs': [0, 1]
        },
        {
            'model_name': 'pvt_v2_b3.in1k', 
            'freeze_mode': 'gradual',
            'unfreeze_epochs': [0, 1]
        },
        {
            'model_name': 'vit_base_patch16_224.orig_in21k_ft_in1k', 
            'freeze_mode': 'gradual',
            'unfreeze_epochs': [0, 1]
        }
    ],
    'dataset_name': 'mock_data',
    'train_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/mock_data/train',
    'val_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/mock_data/val',
    'results_dir': '/content/drive/MyDrive/caltech_proj/caltech_proj/simple_ft/sft_mock_results',
    'batch_size': 256,
    'num_threads': 8,
    'num_epochs': 2,
    'num_classes': 100, 
    'stage_lrs': [1e-3, 1e-4, 5e-5, 1e-5, 1e-6], 
    'weight_decay': 1e-5,
    'early_stop_patience': 3,
    'min_improvement': 0.001
}