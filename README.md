# MNIST-CNN
![[mnist]](assets/mnist.png)


The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems. It contains 60,000 training images and 10,000 testing images. I am still fixing the validation set for this model.

## Objective

In this repository, I coded a convolutional neural network with three conv layers, two pooling layers, and 2 fc layers (multilayer perceptron). The activation function is ReLU and CUDA is implemented. I ran the model on an NVIDIA Geforce RTX 3050.

## Getting Started
### Python Environment
Download and install Python 3.8 or higher from the [official Python website](https://www.python.org/downloads/)

Optional, but I would recommend creating a venv. For Windows installation:
```
py -m venv .venv
.venv\Scripts\activate
```
For Unix/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```

Now install the necessary AI stack in the venv terminal. These libraries will aid with computational coding, data visualization, accuracy reports, preprocessing, etc. I used pip for this project.
```
pip install numPy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install seaborn
```

For PyTorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You will also need to install the torchvision MNIST dataset, which will be prompted in the terminal when called upon.

For CUDA: I downloaded the CUDA toolkit version 12.5 from the NVIDIA website [here](https://developer.nvidia.com/cuda-downloads). I used the network Windows 11 installation.

### Data Input
To input data from the MNIST data set, use the Torchvision library. Below is the code that transforms and splits the data into three sets of loaders. The validiation set, training set, and testing set. The validition set provides an unbiased evaluation of the model. 

```
# import MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # normalized with MNIST mean and std

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```

### Results
![[loss]](assets/loss.png)

This graph demonstrates the training loss with respect to the epochs. 
I ran the model with 5 epochs. The training and testing accuracy are both 99% (rounded down).

![[matrix]](assets/matrix.png)

This is the confusion matrix for the model's results. Here we can visualize the 99% testing accuracy.

