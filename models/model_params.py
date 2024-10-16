
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timm
from models.bs_model_wrapper import BaseTimmWrapper

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Create the ViT model 
    vit_model = BaseTimmWrapper('vit_base_patch16_224.orig_in21k_ft_in1k', num_classes=256, freeze_mode='gradual', head_epochs=1, stage_epochs=1)

    # Count initial trainable parameters (should only be the head)
    initial_params = count_parameters(vit_model)
    print(f"Initial trainable parameters: {initial_params}")

    # Simulate unfreezing stages
    for stage in range(1, vit_model.unfreeze_state['total_stages']):
        new_stage = vit_model.adaptive_unfreeze()
        if new_stage:
            stage_params = count_parameters(vit_model)
            print(f"Trainable parameters after stage {stage}: {stage_params}")
        else:
            print(f"No change in stage {stage}")

    # Full fine-tuning (all parameters)
    vit_model.full_finetune()
    final_params = count_parameters(vit_model)
    print(f"Total parameters when fully fine-tuned: {final_params}")

if __name__ == "__main__":
    main()
