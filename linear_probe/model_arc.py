import timm
import torch

def inspect_model(model_name):
    print(f"\nInspecting model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    
    print(f"Model type: {type(model).__name__}")
    
    # Check for 'fc' attribute
    if hasattr(model, 'fc'):
        print(f"'fc' attribute found: {model.fc}")
    else:
        print("No 'fc' attribute found")
    
    # Check for 'head' attribute
    if hasattr(model, 'head'):
        print(f"'head' attribute found: {model.head}")
    else:
        print("No 'head' attribute found")
    
    # Print the last few layers of the model
    print("\n layers:")
    for name, module in list(model.named_modules()):
        print(f"{name}: {module}")
    
    # Try to identify the classification head
    last_layer = list(model.modules())[-1]
    if isinstance(last_layer, torch.nn.Linear):
        print(f"\nPossible classification head identified: {last_layer}")
        print(f"Input features: {last_layer.in_features}")
        print(f"Output features: {last_layer.out_features}")

# List model names u want to inspect
models_to_inspect = [
   'vit_base_patch16_224.orig_in21k_ft_in1k'
]
# Inspect each model
for model_name in models_to_inspect:
    inspect_model(model_name)
    print("\n" + "="*50)

print("\nInspection complete.")