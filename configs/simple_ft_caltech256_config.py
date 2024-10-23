SIMPLE_FT_CALTECH256_CONFIG = {
    'model_list' : [
        {
            'model_name': 'resnet50.a1_in1k', 
            'ft_strategy': 'full',
            'head_epochs': None,
            'stage_epochs': None
        },
        {
            'model_name': 'regnety_040.pycls_in1k', 
            'ft_strategy': 'gradual',
            'head_epochs': 5,
            'stage_epochs': 10
        },
        {
            'model_name': 'pvt_v2_b3.in1k', 
            'ft_strategy': 'discriminative',
            'head_epochs': None,
            'stage_epochs': None
        },
        {
            'model_name': 'vit_base_patch16_224.orig_in21k_ft_in1k', 
            'ft_strategy': 'gradual',
            'head_epochs': 5,
            'stage_epochs': 10
        }
    ],
    'dataset_name': 'caltech256',
    'train_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/train',
    'val_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/val',
    'results_dir': '/content/drive/MyDrive/caltech_projj/simple_ft/sft_cal_results',
    'models_dir': '/content/drive/MyDrive/caltech_proj/models',
    'num_classes': 256,
    'batch_size': 64,
    'num_threads': 8,
    'num_epochs': 50,
    'base_lr': 1e-4,
    'weight_decay': 1e-5,
    'early_stop_patience': 8,
    'min_improvement': 0.001
}