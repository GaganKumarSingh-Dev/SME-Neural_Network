import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        # Call the parent class's init method to initialize the base class
        super(AlexNet, self).__init__()

        # First Convolutional Layer with 11x11 filters, stride of 4, and 2 padding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

        # Max Pooling Layer with a kernel size of 3 and stride of 2
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Second Convolutional Layer with 5x5 filters and 2 padding
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)

        # Third Convolutional Layer with 3x3 filters and 1 padding
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        # Fourth Convolutional Layer with 3x3 filters and 1 padding
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)

        # Fifth Convolutional Layer with 3x3 filters and 1 padding
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)

        # First Fully Connected Layer with 4096 output features
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)

        # Second Fully Connected Layer with 4096 output features
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)

        # Output Layer with `num_classes` output features
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        # Pass the input through the first convolutional layer and ReLU activation function
        x = self.pool(F.relu(self.conv1(x)))

        # Pass the output of the first layer through
        # the second convolutional layer and ReLU activation function
        x = self.pool(F.relu(self.conv2(x)))

        # Pass the output of the second layer through
        # the third convolutional layer and ReLU activation function
        x = F.relu(self.conv3(x))

        # Pass the output of the third layer through
        # the fourth convolutional layer and ReLU activation function
        x = F.relu(self.conv4(x))

        # Pass the output of the fourth layer through
        # the fifth convolutional layer and ReLU activation function
        x = self.pool(F.relu(self.conv5(x)))

        # Reshape the output to be passed through the fully connected layers
        x = x.view(-1, 256 * 6 * 6)

        # Pass the output through the first fully connected layer and activation function
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)

        # Pass the output of the first fully connected layer through
        # the second fully connected layer and activation function
        x = F.relu(self.fc2(x))

        # Pass the output of the second fully connected layer through the output layer
        x = self.fc3(x)

        # Return the final output
        return x


alexnet = AlexNet()
print(alexnet)


# CHeck parameters summary
# add the cuda to the mode
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# alexnet.to(device)

#Print the summary of the model
# summary(alexnet, (3, 224, 224))
