
import os
import csv
import time
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# config paths. less long
CALTECH_PROJ_PATH = '/content/drive/MyDrive/caltech_proj'
M_CHECKPOINTS_DIR = os.path.join(CALTECH_PROJ_PATH, 'models')
RESULTS_DIR = os.path.join(CALTECH_PROJ_PATH, 'linear_probe', 'lp_results')

os.makedirs(M_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Iniatize models add classification head
class TimmModelWrapper(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.data_config = timm.data.resolve_model_data_config(self.base_model)
        self.input_size = self.data_config['input_size'][1:]
        # Freeze weights.
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.model_name = model_name
        # locate current class head for each model 
        if hasattr(self.base_model, 'head') and hasattr(self.base_model.head, 'fc'): # swin/convnext only ones with .head.fc
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
            raise ValueError(f"Unsupported {model_name}")
        # Add new one 
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.base_model(x) 
        return self.fc(features)
    
    def get_config(self):
        return self.data_config

# Dali Pipeline create     
# is_training to differentiate train/val
@pipeline_def
def create_dali_pipeline(data_dir, data_config, is_training, input_size):
    # want to use model specific config so need to initialize the values.
    crop_pct = data_config['crop_pct']
    interpolation = data_config['interpolation']
    mean = data_config['mean']
    std = data_config['std']

    resize_size = [int(input_size[0] / crop_pct), int(input_size[1] / crop_pct)]

    # Images and labels
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=is_training,
        name="Reader"
    )
    
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB) # mixed to use GPU

    images = fn.resize(images,
                       device="gpu",
                       size=resize_size,
                       mode="not_smaller",
                       interp_type=types.INTERP_CUBIC if interpolation == 'bicubic' else types.INTERP_LINEAR)

    images = fn.crop_mirror_normalize(
        images,
        device='gpu',
        crop=(input_size),
        mean=mean,
        std=std,
        dtype=types.FLOAT,
        mirror=is_training,
        crop_pos_x=0.5,  # center crop
        crop_pos_y=0.5
    )
    return images, labels.gpu()

def get_dali_loader(data_dir, batch_size, num_threads, device_id, data_config, is_training=True):
    input_size = data_config['input_size'][-2:] # [3, 224, 224] why [-2]
    pipeline = create_dali_pipeline(
        data_dir=data_dir,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        data_config=data_config,
        is_training=is_training, #
        input_size=input_size
    )
    pipeline.build()
    return DALIGenericIterator(
        [pipeline],
        ['image', 'label'],
        reader_name="Reader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL # Partial so uses last batch not drop.
    )

def train_and_evaluate(model_wrapper, train_dir, val_dir, batch_size, num_threads, device, num_epochs, data_config):
    model = model_wrapper.to(device)
    optimizer = Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1) # Basic 
    criterion = nn.CrossEntropyLoss()

    model_name = model.model_name

    train_loader = get_dali_loader(train_dir, batch_size, num_threads, 0, data_config, is_training=True)
    val_loader = get_dali_loader(val_dir, batch_size, num_threads, 0, data_config, is_training=False)

    checkpoint_dir = os.path.join(M_CHECKPOINTS_DIR, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epoch_times = []
    best_val_acc = 0.0 

    for epoch in range(num_epochs):
        e_start_time = time.time()

        # Train
        model.train()
        train_loss = 0.0 
        train_correct = 0
        train_total = 0
        for data in train_loader:
            images = data[0]['image']
            labels = data[0]['label'].squeeze(-1).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = train_correct / train_total
        train_accs.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in val_loader:
                images = data[0]['image']
                labels = data[0]['label'].squeeze(-1).long()

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = val_correct / val_total
        val_accs.append(val_accuracy)

        # To save model checkpoint/ need best as base.
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'val_acc': val_accuracy
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'best model_{model_name}.pth')
            torch.save(checkpoint_state, checkpoint_path)
            
        scheduler.step()

        epoch_time = time.time() - e_start_time
        epoch_times.append(epoch_time)

        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f},'
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f},'
              f'Time: {epoch_time:.2f}s')


    return train_losses, train_accs, val_losses, val_accs, epoch_times, best_val_acc

def linear_probing(models, train_dir, val_dir, batch_size, num_threads, device, num_epochs):
    results = []
    csv_file = os.path.join(RESULTS_DIR, 'lprobe_results.csv')

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Validation Acc', 'Epoch Time', 'Cumul Time'])

    # loop though model list 
    for model_name, num_classes in models:
        print(f"Training and evaluating {model_name}")
        model = TimmModelWrapper(model_name, num_classes)
        data_config = model.get_config()

        start_time = time.time()
        t_losses, t_accs, v_losses, v_accs, epoch_times, best_val_acc = train_and_evaluate(model, train_dir, val_dir, batch_size, num_threads, device, num_epochs, data_config)
        total_time = time.time() - start_time

        best_epoch = v_accs.index(max(v_accs)) + 1

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for epoch in range(num_epochs):
                writer.writerow([
                    model_name,
                    epoch + 1,
                    t_losses[epoch],
                    t_accs[epoch],
                    v_losses[epoch],
                    v_accs[epoch],
                    epoch_times[epoch],
                    sum(epoch_times[:epoch+1])
                ])
            # Summary row
            writer.writerow([
                f"{model_name} (Best)",
                best_epoch,
                t_losses[best_epoch-1],
                t_accs[best_epoch-1],
                v_losses[best_epoch-1],
                best_val_acc,
                epoch_times[best_epoch-1],
                sum(epoch_times[:best_epoch])
            ])
        
        results.append((model_name, best_val_acc))
        print(f"\nTraining completed for {model_name}")
        print(f"Final validation accuracy: {best_val_acc:.4f}")
        print(f"Total training time: {total_time:.2f} seconds")
        print("---")

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, num_epochs+1), t_losses, label='Train Loss')
        plt.plot(range(1, num_epochs+1), v_losses, label='Validation Loss')
        plt.title(f'Learning Curves - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f'lr_curves_{model_name.split(".")[0]}.png'))
        plt.close()
        print("Model figure saved")


    return results

# Usage
def main():

    model_list = [ # Trained on Imagenet22k pref if not 1k.
    ('pvt_v2_b3.in1k', 256), #pvt medium 224x
    ('vit_base_patch16_224.orig_in21k_ft_in1k', 256),#ViT 224x
    ('deit3_base_patch16_224.fb_in22k_ft_in1k', 256),  #DeiT 224x
    ('resnet50.a1_in1k', 256),#resnet 50 
    ('resnet18.a1_in1k', 256), #resnet 18
    ('resnet152.a1_in1k', 256), #resnet152
    ('swin_base_patch4_window7_224.ms_in22k_ft_in1k', 256), #swin 224x
    ('mixer_b16_224.goog_in21k_ft_in1k', 256), # MLP-Mixer 224
    ('tf_efficientnetv2_m.in21k_ft_in1k', 256), # Efficient net
    ('convnext_base.fb_in22k_ft_in1k', 256), #Convnext 
    ('regnety_040.pycls_in1k', 256) # Regnet 224
]
    train_dir = '/content/drive/MyDrive/caltech_proj/data/Caltech256_Split/train'
    val_dir = '/content/drive/MyDrive/caltech_proj/data/Caltech256_Split/val'
    batch_size = 128
    num_threads = 8
    num_epochs = 20 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Starting Experiment...")
    start_time = time.time()
    results = linear_probing(model_list, train_dir, val_dir, batch_size, num_threads, device, num_epochs)
    end_time = time.time() - start_time
    print(f"\nTotal experiment time: {end_time:.2f}s")

    print("\nFinal Results:")
    model_names = []
    accs = []
    for model_name, accuracy in results:
      print(f"{model_name}: Accuracy = {accuracy:.4f}")
      model_names.append(model_name.split('.')[0])
      accs.append(accuracy)

    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accs)
    plt.title('Model Performance')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_acc_comparison.png'))
    
    #Get top 3 results 
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    top3_models = sorted_results[:3]
    print("\nTop 3 performing models")
    for i, (model_name, accuracy) in enumerate(top3_models, 1):
      print(f"{i}. {model_name}, Accuracy: {accuracy:4f}")

    #Worst model
    worst_model = sorted_results[-1]
    print(f"\nWorst Performing Model: {worst_model[0]}, Accuracy: {worst_model[1]:.4f}")
    
if __name__ == '__main__':
    main()