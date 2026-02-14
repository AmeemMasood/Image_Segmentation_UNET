 🐶🐱 Image Segmentation using Custom U-Net
📌 Project Overview

This project implements an image segmentation model using a custom U-Net architecture built with PyTorch. The model is trained on the Oxford-IIIT Pet Dataset to perform pixel-wise segmentation of pets from images.

All components including dataset loading, model definition, training, evaluation, and visualization are implemented inside a single Jupyter Notebook (.ipynb).

Project Structure
Image-Segmentation-UNet/
├── segmentation.ipynb   # Main notebook (dataset, model, training, evaluation)
└── README.md   


This project uses the Oxford-IIIT Pet Dataset, which contains images of cats and dogs along with pixel-level segmentation masks.
The dataset is loaded using PyTorch’s built-in API:

> torchvision.datasets.OxfordIIITPet
> Automatic download in Colab enviroment
> Include segmentation annotations

🧠 Model Architecture

The segmentation model is based on a Custom U-Net architecture with the following features:

🔹 Key Components
1.Encoder-Decoder structure
2.Skip connections using concatenation
3.Residual-style feature reuse
4.Convolution + BatchNorm + ReLU blocks
5.Upsampling using transpose convolutions

🔹 Architecture Highlights
1.Downsampling path for feature extraction
2.Bottleneck layer for deep representation
3.Upsampling path for spatial recovery
4.Skip connections to preserve fine details

This design helps the model capture both global context and local details.


⚙️ Technologies Used
1.PyTorch
2.TorchVision
3.NumPy
4.Matplotlib
5.Google Colab


🏋️ Training
1.The training pipeline includes:
2.Data loading and augmentation
3.Loss function: Cross Entropy Loss
4.Optimizer: AdamW
5.Batch-wise training loop
6.Validation after each epoch


📈 Evaluation
Model performance is evaluated using:
1.Training and validation loss curves
2.Visual comparison of predicted masks vs ground truth
3.Sample inference outputs

Evaluation helps verify segmentation accuracy and generalization.


🖼️ Results

After training, the model produces segmentation masks that accurately isolate pets from the background.
Sample outputs include:
Input image
Ground truth mask
Predicted mask

