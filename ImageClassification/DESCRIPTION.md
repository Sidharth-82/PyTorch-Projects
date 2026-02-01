# 🐾 PyTorch Image Classification Project

This repository contains two robust implementations for image classification using **PyTorch**. The project demonstrates both building a custom Convolutional Neural Network (CNN) from scratch and leveraging transfer learning with high-performance pretrained models like GoogleNet.

---

## 🚀 Description

This project focuses on the fundamental task of image classification across different domains:

* **Custom CNN (`image_classification.py`)**: A project specifically designed to classify animal faces (Cats, Dogs, Wild animals) using a custom-built architecture.
* **Transfer Learning (`pretrained_image_classification.py`)**: Utilizes a pretrained GoogleNet (Inception v1) model to classify bean leaf lesions, demonstrating how to adapt state-of-the-art models for specific agricultural disease detection.

---

## 🛠️ Pipeline

Both scripts follow a standardized deep learning pipeline:

1. **Data Acquisition**: Datasets are automatically downloaded from Kaggle using the `opendatasets` library.
2. **Preprocessing**:
* Images are resized to  pixels.
* Normalization is applied using ImageNet standards (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).


3. **Model Architecture**:
* **Custom**: Multi-layer CNN with `ReLU` activations, `MaxPool2d` for downsampling, and `Dropout` for regularization.
* **Pretrained**: Modification of the final fully connected layer of GoogleNet to match the number of target classes.


4. **Training**: Uses the **Adam Optimizer** and **CrossEntropyLoss**. The training loop tracks both training and validation loss/accuracy.
5. **Evaluation & Inference**: Includes functions to load saved model weights (`.pth` files) and run predictions on new, unseen images.

---

## 📖 How to Use

### Prerequisites

Ensure you have the following installed:

* Python 3.x
* PyTorch & TorchVision
* Scikit-Learn (for `LabelEncoder`)
* Pandas & NumPy
* Matplotlib (for progress visualization)
* `torchsummary` (for model architecture visualization)

### Running the Project

1. **Clone the Repository**:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

```


2. **Train the Model**: Run either script to begin training.
```bash
python image_classification.py
# OR
python pretrained_image_classification.py

```


3. **Predict an Image**: Use the `predict_image` function provided within the scripts to test your own images.
