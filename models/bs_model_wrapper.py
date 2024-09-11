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

        self.patience = patience 
        self.min_improvement = min_improvement
        self.state = {
            'best_performance': float('-inf'),
            'epochs_no_imprv': 0,
            'current_stage': 0,
            'best_model_state': None,
            'best_stage': 0 
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
        {'params': [self.base_model.cls_token, self.base_model.pos_embed], 'name': 'embeddings'},
        {'params': self.base_model.patch_embed.parameters(), 'name': 'patch_embed'},
        {'params': attn, 'name': 'attention'},
        {'params': ffn, 'name': 'FFN'},
        {'params': other_params, 'name': 'other_block_params'}
    ]
        return param_group
    
    def get_pvt_params(self):
    
        param_groups = [
            {'params': self.base_model.head.parameters(), 'name': 'head'},
            {'params': self.base_model.patch_embed.parameters(), 'name': 'patch_embed'}
        ]

        for i, stage in enumerate(self.base_model.stages):
            param_groups.append({
                'params': stage.parameters(),'name': f'stage{i}'
            })
        return param_groups
    
    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def adaptive_unfreeze(self, epoch, total_epochs, performance_metric=None):
        if self.freeze_mode != 'gradual':
            return
        
        classifier_epochs = 5
        if epoch < classifier_epochs:
            for group in self.param_groups:
                if group['name'] == 'head':
                        for param in group['params']:
                            param.requires_grad = True
            print(f"Epoch {epoch}: Only classifier unfrozen")
            return
        
        epoch = epoch - classifier_epochs
        new_total_epochs = total_epochs - classifier_epochs

        if 'resnet50' in self.model_name or 'regnety' in self.model_name:
            return self.unfreeze_cnn(epoch, new_total_epochs, performance_metric)
        elif 'vit_base' in self.model_name:
            return self.unfreeze_vit(epoch, new_total_epochs, performance_metric)
        elif 'pvt_v2' in self.model_name:
            return self.unfreeze_pvt(epoch, new_total_epochs, performance_metric)
        else:
            raise ValueError(f"Error in adaptive freeze. {self.model_name}")
        
    def unfreeze_cnn(self, epoch, total_epochs, performance_metric):
        stages = len(self.param_groups) - 1 
        unfreeze_epoch = total_epochs // (stages + 1) # when head trains 1 stage all frozen

        if epoch >= unfreeze_epoch:
            stage_to_unfreeze = min((epoch // unfreeze_epoch) -1, stages -1)
            for i in range(stage_to_unfreeze, -1, -1):
                self.param_groups[i]['requires_grad'] = True
    
    def unfreeze_vit(self, epoch, total_epochs, performance_metric):

        unfreeze_schedule = [
            (0.0, 'head'),
            (0.2, 'FFN'),
            (0.4, 'attention'),
            (0.6, 'other_block')
            (0.7, 'embeddings')
        ]
        current_progress = epoch / total_epochs

        for threshold, param_group_name in unfreeze_schedule:
            if current_progress >= threshold:
                for group in self.param_groups:
                    if group['name'] == param_group_name:
                        for param in group['params']:
                            param.requires_grad = True
                        break
            
        for group in self.param_groups:
            if group['name'] == 'patch_embed':
                for param in group['params']:
                    param.requires_grad = False
                break 
    
    def unfreeze_pvt(self, epoch, total_epochs, performance_metric):
        
        num_stages = len(self.base_model.stages)
        epochs_per_unfreeze = max(1, total_epochs // (num_stages + 1))  # +1 for patch_embed

        current_unfreeze = min(epoch // epochs_per_unfreeze, num_stages)

        parts_to_unfreeze = [f'stage{i}' for i in range(num_stages-1, -1, -1)] + ['patch_embed']

        for i, part in enumerate(parts_to_unfreeze):
            if i <= current_unfreeze:
                for group in self.param_groups:
                    if group['name'].startswith(part):
                        for param in group['params']:
                            param.requires_grad = True

        print(f"Adjusted Epoch {epoch}: Unfreezing up to {parts_to_unfreeze[current_unfreeze]}")
        
        
                            

        
    


