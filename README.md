# Model Comparison on Caltech256 & Cifar100

## Project Overview

This project conducts a study of how pre-trained models perform on the task of image classification using the Caltech256 & Cifar100 dataset. We explore which model architecture inherently captures the features of these datasets most effectively, and then fine-tune these models to achieve maximum performance. The generalization capabilities of the models will then be tested on a small subset of the Open Images Dataset. 

## Motivation 

Both these datasets were picked due to their stark differences and popularity. The caltech256 emulates real-world scenarios so has more noise than the cifar100, this means images have variations in background, lighting and angle. Caltech256 also has varying image resolutions with the most common size being 640x480 in the val set for example. Whereas in cifar100 all images are 32x32 so is a controlled uniformed dataset. Therefore, by using both datasets we can understand which architectures are most suitable for different use cases.

The questions that this study tries to answer are:

- Which model architecture inherently captures features of the datasets most effectively (best feature extractor)?
- Which model demonstrates versatility across both high and low resolution images?
- How does a simple CNN perform compared to fine-tuned pre-trained models?
- How well do the models generalize to unseen data?

## Methodology

Our approach has four main experimental phases:

1.  **Linear probing**: Evaluating the feature extraction captabilities of models.
2.  **Simple fine-tuning**: Assessing how well the chosen models adapt when weights are unfrozen.
3.  **Full fine-tuning**: Maximizing performance of best model(s) from the simple fine-tuning phase. 
4. **Generalization**: Testing generalization capabilites of both pre-trained models and the simple CNN.

## Datasets

Datasets | Caltech256 | Cifar100 | Open Image subset |
|--------|------------|----------|-------------------| 
|Train: | 20,729 | 43,750 |    |
|Val:  | 2,971 | 6,250 |        |
|Test: | 6080 | 10,000 |        |
|Total: | 29,780 |  60,000 |  TODO  |

- [Link to Caltech256 dataset](https://data.caltech.edu/records/nyy15-4j048)
- [Link to Cifar100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Linear Probing 

### Key Findings 

Here are the following results from the linear probing experiment.

 Model | Caltech256 |   Cifar100 |  
|-------|-----------|----------------------|
| PVT V2 B3 | 77.68% |              |
| ViT Base | 15.38% |               |
| DeiT3 Base | 13.93% |             |
| ResNet50 | 82.90% |               |
| ResNet18 | 75.80% |               |
| ResNet152 | 80.58% |              |
| Swin Base | 74.28% |              |
| MLP-Mixer | 6.73%  |              |
| EfficientNetV2 M  | 93.20%|       |
| ConvNext Base | 82.30 |           |
| Regnety_040 | 87.11% |            |

![Model Performance Comparison](./linear_probe/lp_results/model_acc_comparison.png)

### Key Observations:

1. EfficientNetV2 was the best perfoming feature extractor with an outstanding 93.2% accuracy, the model inherently captures the features of Caltech256 best.
2. The worst performing model is the MLP-mixer with 6.73% which is normal as there is no/little inductive bias, just fully connected layers essentially.
3.  All models which use convolution for embedding  perform signicantly better than models which uses patches (vit/deit3). Therefore, the hierarchical representation of an image is vital for a feature extractor.
4. Transformers need to be fine-tuned on the dataset in order to be effective, the global & high level features learned are not trasferable to the new task unlike with CNN's which capture more local features.

## Simple Fine-tuning

The following models will be picked for the simple fine-tuning: 

1. EfficientNetV2: Best performing model (CNN)
2. MLP-Mixer: Worst model, let's see how well performs when trained.
3. PVT V2 B3: Hybrid model
4. Deit3: Transformer model.

## Setup and Installation 





