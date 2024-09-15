import timm 
from bs_model_wrapper import BaseTimmWrapper

model = 'resnet50.a1_1nk'
unfreeze_epochs = [0, 5, 10, 15, 20, 25]
resnet50 = BaseTimmWrapper(model, 100, freeze_mode='gradual', unfreeze_epochs=unfreeze_epochs)


