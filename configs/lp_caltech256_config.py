LP_CALTECH256_CONFIG = {
    'model_list': [ 
    ('pvt_v2_b3.in1k', 256), 
    ('vit_base_patch16_224.orig_in21k_ft_in1k', 256),
    ('deit3_base_patch16_224.fb_in22k_ft_in1k', 256),  
    ('resnet50.a1_in1k', 256),
    ('resnet18.a1_in1k', 256),
    ('resnet152.a1_in1k', 256),
    ('swin_base_patch4_window7_224.ms_in22k_ft_in1k', 256),
    ('mixer_b16_224.goog_in21k_ft_in1k', 256),
    ('tf_efficientnetv2_m.in21k_ft_in1k', 256),
    ('convnext_base.fb_in22k_ft_in1k', 256),
    ('regnety_040.pycls_in1k', 256) 
    ],
    'train_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/train',
    'val_dir': '/content/drive/MyDrive/caltech_proj/data/Caltech256/val',
    'batch_size': 256,
    'num_threads': 8,
    'num_epochs': 5,
    'results_dir': '/content/drive/MyDrive/caltech_proj/caltech_proj/linear_probe/lp_cal_results'
}