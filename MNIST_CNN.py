import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

#enables CUDA
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Params
num_epochs = 5
batch_size = 100
learning_rate = 0.001
train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # normalized with MNIST mean and std

train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# CNN (two layer)
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels= 16, kernel_size= 5, stride = 1, padding = 2),
            nn.ReLU())
        self.fc1 = nn.Linear(7 * 7 * 64, 32) # 10 is the output, we need 10 for the 10 digits of MNIST. 64 is the number of channels
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet().to(device) # declare model as a ConvNet

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train function
def train(model, criterion, optimizer, train_loader, num_epochs):
    model.train() # sets into train mode

    for epoch in range(num_epochs):
        total_train_loss = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            optimizer.zero_grad() # clears previous gradients
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprop and optimize
            loss.backward()
            optimizer.step()

            # Sum up total loss
            total_train_loss += loss.item()

            # Count correct predictions
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        # Calculates the average train loss and accuracy
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses[epoch] = avg_train_loss
        train_accuracy = 100 * correct / len(train_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.5f}, Train Accuracy: {train_accuracy:.3f}%")

# Test function
def test(model, criterion, test_loader, num_epochs):
    model.eval()  # eval mode
    with torch.no_grad():
        for epoch in range(num_epochs):
            total_test_loss = 0
            correct = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                #Forward Pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Sums up the total loss
                total_test_loss += loss.item()

                # Counts the number of correct predictions
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            avg_test_loss = total_test_loss / len(test_loader)
            test_losses[epoch] = avg_test_loss
            test_accuracy = 100 * correct / len(test_loader.dataset)

    print(f"Test Loss: {avg_test_loss:.5f}%")

    print(f"Test Accuracy: {test_accuracy:.3f}%")

# Train and test the model
train(model, criterion, optimizer, train_loader, num_epochs)
test(model, criterion, test_loader, num_epochs)

# Plot the losses over time
plt.plot(train_losses, label='Train Loss')
plt.legend()
plt.ylabel('Train Loss')
plt.xlabel('Epochs')
plt.savefig('MNIST-CNN/assets/loss.png') # saves 
plt.show()


# Confusion matrix
y_pred = []
y_true = []

# iterate over test data   
for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Correct Ones

# constant for classes
classes = ('0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sns.heatmap(df_cm, annot=True)
plt.savefig('MNIST-CNN/assets/matrix.png') # saves confusion matrix as png to assets
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'testModel.ckpt')
