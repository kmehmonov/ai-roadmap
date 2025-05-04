import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: reimplement
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # )
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2) # for input size 3x224x224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(9216, 9216)
        self.fc2 = nn.Linear(9216, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 1000)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = F.relu(self.maxpool1(x))
        print(x.shape)

        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.relu(self.maxpool2(x))
        print(x.shape)

        x = F.relu(self.conv3(x))
        print(x.shape)

        x = F.relu(self.conv4(x))
        print(x.shape)

        x = F.relu(self.conv5(x))
        print(x.shape)
        
        x = F.relu(self.maxpool3(x))
        print(x.shape)

        x = self.flat(x)

        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = F.relu(self.fc3(x))
        print(x.shape)
        x = self.fc4(x)
        print(x.shape)
        return x

      
if __name__ == '__main__':
    model = AlexNet()
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
