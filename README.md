# 📄 Supervised object localization using vision transformer and Visual Foundation Models:  


## 📌 Abstract  
Weakly-supervised object localization is the task of identifying the objects in images with bounding boxes when only image class labels are available during the training.  In this work, we leverage the dense features extracted from pre-trained Visual Foundation Models (VFMs) and train a transformer network to predict foreground probability distributions. The feature of the proposed method is to disentangle the classification path from the segmentation path through the frozen and pretrained features in VFMs, preventing the map from focusing on discriminative parts of the image. We use momentum contrastive learning (MoCo) to classify the patch tokens as foreground and background in a self-supervised manner. The contrastive loss is designed to increase the distance between the a global representation of foreground and background features of the image. The experiments show state-of-the-art performance of our method in terms of localization accuracy compared to other methods including the ones that use image labels.

## 🌟 Highlights  
- ✅  State-of-the-art method for weakly supervised object localization based on ViTs
- ✅  The method is completely based on visual features and unlike other methods does not require heuristic text prompts
- ✅  Only a lightweight transformer is trained resulting in a fast training

## 🖼️ Method Overview  
![Caption for the image](path/to/image.png)  
*Figure 1: Brief caption describing the image.*  

