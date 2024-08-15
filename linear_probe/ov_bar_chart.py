import matplotlib.pyplot as plt 
import numpy as np 

models = ['EfNetV2-M', 'RegNetY-040', 'Resnet50', 'CvNext_Base', 'Resnet152', 'pvt_v2_B3', 'Resnet18', 'swin_base', 'vit_base', 'deit3_base', 'mlp-mixer']
caltech256_acc = [93.00, 85.86, 82.09, 82.30, 78.69, 77.18, 75.73, 73.75, 14.47, 12.22, 6.46]
cifar100_acc = [64.78, 62.64, 51.89, 51.71, 52.59, 50.59, 44.71, 26.85, 18.77, 16.69, 8.96]

x = np.arange(len(models))
width = 0.3

fig, ax = plt.subplots(figsize=(12, 6))
bar1 = ax.bar(x - width/2, caltech256_acc, width, label='caltech256', color='lightblue')
bar2 = ax.bar(x + width/2, cifar100_acc, width, label='cifar100', color='darkblue')

ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()

ax.bar_label(bar1, padding=3, rotation=90)
ax.bar_label(bar2, padding=3, rotation=90)

fig.tight_layout()

plt.savefig('caltech_proj/linear_probe/lp_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()