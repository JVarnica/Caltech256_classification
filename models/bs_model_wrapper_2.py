# base_model.py
import sys
sys.path.append(sys.path[0] + "/..")
import timm
import torch.nn as nn

class BaseTimmWrapper(nn.Module):
    def __init__(self, model_name, num_classes, ft_strategy=None, head_epochs=None, stage_epochs=None):
        super().__init__()
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.data_config = timm.data.resolve_model_data_config(self.base_model)
        self.input_size = self.data_config['input_size'][1:] 
        self.model_name = model_name.split('.')[0]
        self.ft_strategy = ft_strategy 
        self.head_epochs = head_epochs
        self.stage_epochs = stage_epochs
        self.epochs_in_current_stage = 0
        self.unfrozen_param_count = []

        self.setup_classifier(num_classes)
        self.setup_fine_tuning()

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
        
    def setup_fine_tuning(self):
        if self.ft_strategy == "gradual":
            self.freeze_all_params()
            self.unfreeze_classifier()
        elif self.ft_strategy in ["discriminative", "full"]:
            self.unfreeze_all_params()
        else:
            raise ValueError(f"Unsurpoorted Finetuning Strategy!! {self.ft_strategy}")
        
        if self.ft_strategy != "discriminative":
            self.param_groups = self.get_param_groups

        self.param_groups = self.get_param_groups()
        self.stages = [group['name'] for group in self.param_groups]
        self.current_stage = 0
        self.total_stages = len(self.param_groups)

    def freeze_all_params(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_all_params(self):
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def unfreeze_classifier(self): #Unfreeze all params in head drop_out, pooling etc.
        if hasattr(self.base_model, 'fc'):
            for param in self.base_model.fc.parameters():
                param.requires_grad = True
        if hasattr(self.base_model, 'head'):
            for param in self.base_model.head.parameters():
                param.requires_grad = True

    def get_param_groups(self): #Model params top down order
        params = []

        lr_mult_base = 0.1 if self.ft_strategy == "discriminative" else 1.0
        if 'resnet' in self.model_name:

            params.append({'name': 'head', 'params': self.base_model.fc.parameters(), 'lr_mult': 1.0}) # Head

            layer4_params = list(self.base_model.layer4.parameters()) 
            if hasattr(self.base_model, 'global_pool'):
                layer4_params.extend(self.base_model.global_pool.parameters())

            params.append({'name': 'stage4', 'params': layer4_params, 'lr_mult': lr_mult_base * 4})
            params.append({'name': 'stage3', 'params': self.base_model.layer3.parameters(), 'lr_mult': lr_mult_base * 3})
            stage1_params = list(self.base_model.layer2.parameters()) + list(self.base_model.layer1.parameters())
            params.append({'name': 'layers2_1', 'params': stage1_params, 'lr_mult': lr_mult_base * 2})
        
            embed_params = []
            for layer in ['conv1', 'bn1', 'relu', 'maxpool']:
                if hasattr(self.base_model, layer):
                    embed_params.extend(getattr(self.base_model, layer).parameters())
            params.append({'name': 'embeddings', 'params': embed_params, 'lr_mult': lr_mult_base})

        elif 'regnety' in self.model_name:
            params.append({'name': 'head', 'params': self.base_model.head.parameters(), 'lr_mult': 1.0})
            params.append({'name': 'stage4', 'params': self.base_model.s4.parameters(), 'lr_mult': lr_mult_base * 4})
            params.append({'name': 'stage3', 'params': self.base_model.s3.parameters(), 'lr_mult': lr_mult_base * 3})

            stage1_params = list(self.base_model.s2.parameters()) + list(self.base_model.s1.parameters())
            params.append({'name': 'stages2_1', 'params': stage1_params, 'lr_mult': lr_mult_base * 2})
            
            params.append({'name': 'stem', 'params': self.base_model.stem.parameters(), 'lr_mult': lr_mult_base})

        elif 'vit_base' in self.model_name:
            params.append({'name': 'head', 'params': self.base_model.head.parameters(), 'lr_mult': 1.0})

            stage4_params = [p for block in self.base_model.blocks[8:] for p in block.parameters()] + list(self.base_model.norm.parameters()) + list(self.base_model.fc_norm.parameters())
            params.append({'name': 'stage4', 'params': stage4_params,'lr_mult': lr_mult_base * 4})

            params.append({'name': 'stage3', 'params': [p for block in self.base_model.blocks[4:8] for p in block.parameters()],
                           'lr_mult': lr_mult_base * 3})
            params.append({'name': 'stage2', 'params': [p for block in self.base_model.blocks[:4] for p in block.parameters()],
                           'lr_mult': lr_mult_base * 2})
            embed = list(self.base_model.patch_embed.parameters()) + list(self.base_model.norm_pre.parameters())
            embed.extend([self.base_model.cls_token, self.base_model.pos_embed])
            params.append({'name': 'embeddings', 'params': embed, 'lr_mult': lr_mult_base})

        elif 'pvt' in self.model_name:
            params.append({'name': 'head', 'params': self.base_model.head.parameters(), 'lr_mult': 1.0})
            params.append({'name': 'stage4', 'params': self.base_model.stages[3].parameters(),
                           'lr_mult': lr_mult_base * 4})
            params.append({'name': 'stage3', 'params': self.base_model.stages[2].parameters(),
                           'lr_mult': lr_mult_base * 3})
            stage2_params = list(self.base_model.stages[1].parameters()) + list(self.base_model.stages[0].parameters())
            params.append({'name': 'stage2', 'params': stage2_params,
                           'lr_mult': lr_mult_base * 2})
            params.append({'name': 'embeddings', 'params': self.base_model.patch_embed.parameters(), 'lr_mult': lr_mult_base})
            
        return params

    def adaptive_unfreeze(self, force_unfreeze=False):
        
        if self.ft_strategy == 'gradual':
            if self.current_stage < self.total_stages:
                if force_unfreeze or self.epochs_in_current_stage >= self.stage_epochs:
                    self.current_stage += 1
                    for param in self.param_groups[self.current_stage]['params']:
                        param.requires_grad = True
                    self.epochs_in_current_stage = 0
                    return True
        elif self.ft_strategy == 'discriminative':
            if force_unfreeze:
                for group in self.param_groups:
                    group['lr_mult'] *= 0.1
                return True
        return False       
    
    def full_finetune(self):
        self.freeze_mode = 'none'
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.current_stage = self.total_stages
        return True
    
    def forward(self, x):
        return self.base_model(x)

    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.base_model.parameters())
    
    def get_config(self):
        return self.data_config
    
        
        
        
        

        
                            

        
    


