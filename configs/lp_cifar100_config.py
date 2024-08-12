CIFAR100_CONFIG = {
    'model_list': [ 
    ('pvt_v2_b3.in1k', 100), 
    ('vit_base_patch16_224.orig_in21k_ft_in1k', 100),
    ('deit3_base_patch16_224.fb_in22k_ft_in1k', 100),  
    ('resnet50.a1_in1k', 100),
    ('resnet18.a1_in1k', 100),
    ('resnet152.a1_in1k', 100),
    ('swin_base_patch4_window7_224.ms_in22k_ft_in1k', 100),
    ('mixer_b16_224.goog_in21k_ft_in1k', 100),
    ('tf_efficientnetv2_m.in21k_ft_in1k', 100),
    ('convnext_base.fb_in22k_ft_in1k', 100),
    ('regnety_040.pycls_in1k', 100) 
    ],
    'train_dir': '/content/drive/MyDrive/caltech_proj/data/Cifar100/train',
    'val_dir': '/content/drive/MyDrive/caltech_proj/data/Cifar100/val',
    'batch_size': 128,
    'num_threads': 8,
    'num_epochs': 5,
    'results_dir': '/content/drive/MyDrive/caltech_proj/caltech_proj/linear_probe/lp_cf_results'
}