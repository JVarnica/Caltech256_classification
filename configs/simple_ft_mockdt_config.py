SIMPLE_FT_MOCK_DATA_CONFIG = {
    'model_list' : [
        ('resnet50.a1_in1k', 256, 'gradual'),
        ('regnety_040.pycls_in1k', 256, 'gradual'),
        ('mixer_b16_224.goog_in21k_ft_in1k', 256, 'none'),
        ('pvt_v2_b3.in1k', 256, 'gradual'),
        ('vit_base_patch16_224.orig_in21k_ft_in1k', 256, 'gradual')
    ],
    'train_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/mock_data/train',
    'val_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/mock_data/val',
    'batch_size': 128,
    'num_threads': 8,
    'num_epochs': 2,
    'results_dir': '/content/drive/MyDrive/caltech_proj/caltech_proj/linear_probe/lp_mock_results'
}