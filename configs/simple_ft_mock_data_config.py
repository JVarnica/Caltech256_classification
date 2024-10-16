SIMPLE_FT_MOCK_DATA_CONFIG = {
    'model_list' : [
           {
            'model_name': 'resnet50.a1_in1k', 
            'freeze_mode': 'gradual',
        },
        {
            'model_name': 'regnety_040.pycls_in1k', 
            'freeze_mode': 'gradual',
        },
        {
            'model_name': 'pvt_v2_b3.in1k', 
            'freeze_mode': 'gradual',
        },
        {
            'model_name': 'vit_base_patch16_224.orig_in21k_ft_in1k', 
            'freeze_mode': 'gradual',
        }
    ],
    'dataset_name': 'mock_data',
    'train_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/mock_data/train',
    'val_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/mock_data/val',
    'results_dir': '/content/drive/MyDrive/caltech_proj/simple_ft/sft_mock_results',
    'batch_size': 64,
    'num_threads': 8,
    'num_epochs': 2,
    'num_classes': 256, # caltech mock_data
    'head_epochs': 5,
    'stage_epochs': 10,
    'stage_lrs': [1e-3, 1e-4, 5e-5, 1e-5, 1e-6], 
    'weight_decay': 1e-5,
    'early_stop_patience': 3,
    'min_improvement': 0.001
}