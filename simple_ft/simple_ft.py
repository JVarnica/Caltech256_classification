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
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

def get_exp_config(dataset_name):
    config_module = importlib.import_module(f'configs.{dataset_name}_config')
    return getattr(config_module, f'{dataset_name.upper()}_CONFIG')

