SIMPLE_FT_CALTECH256_CONFIG = {
    'model_list' : [
        ('resnet50.a1_in1k', 256, 'gradual'),
        ('regnety_040.pycls_in1k', 256, 'gradual'),
        ('mixer_b16_224.goog_in21k_ft_in1k', 256, 'none'),
        ('pvt_v2_b3.in1k', 256, 'gradual'),
        ('vit_base_patch16_224.orig_in21k_ft_in1k', 256, 'gradual')
    ],
    'train_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/train',
    'val_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/val',
    'batch_size': 256,
    'num_threads': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'results_dir': '/content/drive/MyDrive/caltech_proj/caltech_proj/simple_ft/sft_cal_results'
}