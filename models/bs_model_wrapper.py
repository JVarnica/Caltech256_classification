# base_model.py
import sys
sys.path.append(sys.path[0] + "/..")
import timm
import torch.nn as nn
import copy

class BaseTimmWrapper(nn.Module):
    def __init__(self, model_name, num_classes, freeze_mode='full', unfreeze_epochs=None):
        super().__init__()
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.data_config = timm.data.resolve_model_data_config(self.base_model)
        self.input_size = self.data_config['input_size'][1:]
        self.model_name = model_name.split('.')[0]
        self.freeze_mode = freeze_mode
        self.set_base_model_state(freeze_mode)
        self.setup_classifier(num_classes)
        self.param_groups = self.get_param_groups()
        self.stages = [group['name'] for group in self.param_groups]
        self.unfreeze_epoch = unfreeze_epochs

        self.unfreeze_state = {
            'current_stage': 0,
            'total_stages': len(self.param_groups),
            'stage_history': [],
            'best_performance': 0
        }
        
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
            return self.get_vit_params(self.base_model)
        elif 'pvt_v2' in self.model_name:
            return self.get_pvt_params(self.base_model)
        elif 'regnety' in self.model_name:
            return self.get_regnety_params(self.base_model)
        else:
            return [{'params' : self.parameters()}]
    
    def get_resnet50_params(self):
        param_group = [
            {'params': self.base_model.fc.parameters(), 'name': 'head'},
            {'params': self.base_model.layer4.parameters(), 'name': 'layer4'},
            {'params': self.base_model.layer3.parameters(), 'name': 'layer3'},
            {'params': self.base_model.layer2.parameters(), 'name': 'layer2'},
            {'params': self.base_model.layer1.parameters(), 'name': 'layer1'},
            {'params': nn.Sequential(self.base_model.conv1, self.base_model.bn1).parameters(), 'name': 'conv1_bn1'}
        ]
        return param_group
    
    def get_regnety_params(self):
        param_group = [
        {'params': self.base_model.head.fc.parameters(), 'name': 'head'},
        {'params': self.base_model.s4.parameters(), 'name': 'block4'},
        {'params': self.base_model.s3.parameters(), 'name': 'block3'},
        {'params': self.base_model.s2.parameters(), 'name': 'block2'},
        {'params': self.base_model.s1.parameters(), 'name': 'block1'},
        {'params': self.base_model.stem.parameters(), 'name': 'stem'}
        ]
        return param_group
    
    def get_vit_params(self):

        attn = []
        ffn = []
        other_params = []
        
        for block in self.base_model.blocks:
            attn.extend(block.attn.parameters())
            ffn.extend(block.mlp.parameters())

            for name, param in block.named_parameters():
                if not any(x in name for x in ['attn', 'mlp']):
                    other_params.append([param])

        param_group = [
        {'params': self.base_model.head.parameters(), 'name': 'head'},
        {'params': ffn, 'name': 'FFN'},
        {'params': attn, 'name': 'attention'},
        {'params': other_params, 'name': 'other_block_params'},
        {'params': [self.base_model.cls_token, self.base_model.pos_embed], 'name': 'embeddings'},
        {'params': self.base_model.patch_embed.parameters(), 'name': 'patch_embed'}
        ]
        return param_group
    
    def get_pvt_params(self):
        param_group = [
            {'params': self.base_model.head.parameters(), 'name': 'head'},
            {'params': self.base_model.patch_embed.parameters(), 'name': 'patch_embed'},
            {'params': self.base_model.stages[3].parameters(), 'name': 'stage 4'},
            {'params': self.base_model.stages[2].parameters(), 'name': 'stage 3'},
            {'params': self.base_model.stages[1].parameters(), 'name': 'stage 2'},
            {'params': self.base_model.stages[0].parameters(), 'name': 'stage 1'}
        ]
        return param_group
    
    def adaptive_unfreeze(self, epoch, performance_metric):
        if self.freeze_mode != 'gradual':
            return False, None
       
        current_stage = next((i for i, e in enumerate(self.unfreeze_epoch) if epoch < e), len(self.stages))
        
        if current_stage > self.unfreeze_state['current_stage']:
            previous_state = self.state_dict().copy() # Save state befr nxt stage 
            print(f"Unfreezing stage {current_stage}: {self.stages[current_stage-1]}")
            self.unfreeze_state['current_stage'] = current_stage
            for i in range(current_stage):
                for param in self.param_groups[i]['params']:
                    param.requires_grad = True

            reason = 'epoch' if epoch in self.unfreeze_epoch else 'performance'
            self.unfreeze_state['state_history'].append((current_stage, reason))
            return True, previous_state
        
        if performance_metric > self.unfreeze_state['best_performance']:
            self.unfreeze_state['best_performance'] = performance_metric

        return False, None
    
    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.parameters())
        
        
        
        

        
                            

        
    


