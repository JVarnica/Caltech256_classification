# base_model.py
import sys
sys.path.append(sys.path[0] + "/..")
import timm
import torch.nn as nn

class BaseTimmWrapper(nn.Module):
    def __init__(self, model_name, num_classes, freeze_mode='full'):
        super().__init__()
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.data_config = timm.data.resolve_model_data_config(self.base_model)
        self.input_size = self.data_config['input_size'][1:]
        self.model_name = model_name.split('.')[0]
        
        self.set_base_model_state(freeze_mode)
        self.setup_classifier(num_classes)
        self.param_groups = self.get_param_groups()

    def set_base_model_state(self, freeze_mode):
        if freeze_mode == 'full':
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
        elif freeze_mode == 'none':
            self.base_model.train()
            for param in self.base_model.parameters():
                param.requires_grad = True
        elif freeze_mode == 'gradual':
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Gradual unfreezing will be handled during training
        else:
            raise ValueError(f"Unsupported freeze mode: {freeze_mode}")

    def setup_classifier(self, num_classes):
        # Keep architecture the same 
        if 'regnety' in self.model_name:
            in_features = self.base_model.head.fc.in_features
            self.base_model.head.fc = nn.Linear(in_features, num_classes)
        elif 'resnet50' in self.model_name:
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)
        elif 'vit_base' in self.model_name:
            in_features = self.base_model.head.in_features
            self.base_model.head = nn.Linear(in_features, num_classes)
        elif 'pvt_v2' in self.model_name:
            in_features = self.base_model.head.in_features
            self.base_model.head = nn.Linear(in_features, num_classes)
        else: 
            raise ValueError(f"Unsupported {self.model_name}")

    def forward(self, x):
        return self.base_model(x)
    
    def get_config(self):
        return self.data_config

    def get_param_groups(self):
        if 'resnet50' in self.model_name:
            return self.get_resnet50_params(self.base_model)
        elif 'vit_base' in self.model_name:
            return self.get_vit_params()
        elif 'pvt_v2' in self.model_name:
            return self.get_pvt_params(self.base_model)
        elif 'regnety' in self.model_name:
            return self.get_regnety_params(self.base_model)
        else:
            return [{'params' : self.parameters()}]
    
    def get_resnet50_params(self):
        return [
               {'params': self.base_model.fc.parameters(), 'name': 'fc'},
            {'params': self.base_model.layer4.parameters(), 'name': 'layer4'},
            {'params': self.base_model.layer3.parameters(), 'name': 'layer3'},
            {'params': self.base_model.layer2.parameters(), 'name': 'layer2'},
            {'params': self.base_model.layer1.parameters(), 'name': 'layer1'},
            {'params': nn.Sequential(self.base_model.conv1, self.base_model.bn1).parameters(), 'name': 'conv1_bn1'}
        ]
    def get_regnety_params(self):
        return [
             {'params': self.base_model.head.fc.parameters(), 'name': 'fc'},
        {'params': self.base_model.s4.parameters(), 'name': 'block4'},
        {'params': self.base_model.s3.parameters(), 'name': 'block3'},
        {'params': self.base_model.s2.parameters(), 'name': 'block2'},
        {'params': self.base_model.s1.parameters(), 'name': 'block1'},
        {'params': self.base_model.stem.parameters(), 'name': 'stem'}
        ]
    
    def get_vit_params(self):
        return [
        {'params': self.base_model.cls_token, 'name': 'cls_token'},
        {'params': self.base_model.pos_embed, 'name': 'pos_embed'},
        {'params': self.base_model.patch_embed.parameters(), 'name': 'patch_embed'},
        {'params': [p for block in self.base_model.bloks for p in block.attn.parameters()], 'name': 'attention'},
        {'params': [p for block in self.base_model.blocks for p in block.mlp.parameters()], 'name': 'FFN'}
        {'params': self.base_model.head.parameters(), 'name': 'head'},
        {'params': self.base_model.norm.parameters(), 'name': 'norm'}
    ]
    
    def get_pvt_params():
        return 
    
    def adaptive_unfreeze():
        return
    


