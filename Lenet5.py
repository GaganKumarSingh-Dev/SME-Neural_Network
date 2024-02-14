import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        # Call the parent class's init method
        super(LeNet5, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # First Fully Connected Layer
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        # Second Fully Connected Layer
        self.fc2 = nn.Linear(in_features=120, out_features=84)

        # Output Layer
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Pass the input through the first convolutional layer and activation function
        x = self.pool(F.relu(self.conv1(x)))
        # Pass the output of the first layer through
        # the second convolutional layer and activation function
        x = self.pool(F.relu(self.conv2(x)))
        # Reshape the output to be passed through the fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        # Pass the output through the first fully connected layer and activation function
        x = F.relu(self.fc1(x))
        # Pass the output of the first fully connected layer through
        # the second fully connected layer and activation function
        x = F.relu(self.fc2(x))
        # Pass the output of the second fully connected layer through the output layer
        x = self.fc3(x)
        # Return the final output
        return x


lenet5 = LeNet5()
print(lenet5)
# add the cuda to the mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lenet5.to(device)

#Print the summary of the model
summary(lenet5, (1, 32, 32))
