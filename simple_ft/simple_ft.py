import os
import argparse
import importlib
import csv
import time
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from linear_probe.model_eval import create_dali_pipeline, get_dali_loader
from models.bs_model_wrapper import BaseTimmWrapper

def get_exp_config(dataset_name):
    config_module = importlib.import_module(f'configs.{dataset_name}_config')
    return getattr(config_module, f'{dataset_name.upper()}_CONFIG')

def model_training(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs):

