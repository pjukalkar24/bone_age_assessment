# bone_age_assessment

### Abstract
To perform BAA, x-ray images must first be segmented to reduce noise and complexity; semantic (supervised) segmentation has delivered promising results2 in predicting bone age. However, current approaches to BAA often utilize supervised convolutional neural networks2,3,4 (CNNs) which, while capable of outperforming radiologists, are limited by data concerns. CNNs require annotated segmentations, labeled by hand pixel-by-pixel as 0 or 1 based on relevance2. To combat this problem of “expensive” and at times unavailable data, our proposed architecture aims to utilize a small yet varied dataset without any need for perfectly annotated segmentations. The proposed architecture is able to use existing x-ray datasets in unique ways, automatically creating data in the form of rough, flawed segmentations from which the neural network can learn to achieve desirable results. By utilizing existing data in unique ways and capitalizing on available resources, the horizons of biomedical imaging can be expanded to include unsupervised learning, which has been largely ignored in favor of semantic segmentation. This study proposes the use of a Dense-Dense U-Net with K-Means derived ground truths to alleviate the need for extensive expert-labelled data while preserving state-of-the-art (SOTA) accuracy in segmentation task.  

# Running this Repository
Data augmentation, full pipeline, and K-Means clustering does not need to be run. Results are reproducible with seed 42 and stored in folders in repository.
UNET_FINAL does not need to be run. Model weights are stored in FINAL_UNET_MODEL.h5.
BoneAgeAssessment must be run to view results.  

# File Breakdown
## Data
data --> train, test, validation --> unprocessed x-ray images
pipeline_outputs --> data --> train, test, validation --> processed x-ray images from FullPipeline.ipynb
kmeans_segm --> data --> k-means clustered x-ray images
BoneAge.csv --> image number, corresponding bone age, and gender

## Data Pre-Processing
DataAugmentation.ipynb --> Introduces noise and randomness to x-ray imaging to mimic differences in x-ray technology across hospitals
FullPipeline.ipynb --> Completes all pre-processing for x-ray images for input
KMeans.ipynb --> Applies K-Means clustering to all x-ray images in training set to be used as pseudo ground truths

## Proposed Model
UNET_FINAL.ipynb --> Runs DDU-Net training for the task of image segmentation **\*WARNING: CPU runtime >5 hours\***
BoneAgeAssessment.ipynb --> Runs SVR, DF, Boosting, and NN models to predict bone age

## Model Weights
DDUNET_train.h5 --> original model weights with standard loss
DDUNET_train_custom_loss[_2].h5 --> model weights with proposed loss
**FINAL_UNET_MODEL.h5 --> model weights for proposed model and loss w/ optimal hyperparameters**

## Logs
logs, logs_ddunet, logs_ddunet_custom_loss[_2], logs_dice --> loss during u-net training to be used with tensorboard
