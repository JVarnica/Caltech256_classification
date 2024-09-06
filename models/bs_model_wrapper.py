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
        self.model_name = model_name
        
        self.set_base_model_state(freeze_mode)
        self.setup_classifier(num_classes)

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
        
        if hasattr(self.base_model, 'head') and hasattr(self.base_model.head, 'fc'):
            in_features = self.base_model.head.fc.in_features
            self.base_model.head.fc = nn.Identity()
        elif hasattr(self.base_model, 'head'):
            in_features = self.base_model.head.in_features
            self.base_model.head = nn.Identity()
        elif hasattr(self.base_model, 'fc'):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif hasattr(self.base_model, 'classifier'):
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        else: 
            raise ValueError(f"Unsupported {self.model_name}")
        # Add new one 
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        return self.fc(features)
    
    def get_config(self):
        return self.data_config

    def gradual_unfreeze(self, num_layers):
        layers = list(self.base_model.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

