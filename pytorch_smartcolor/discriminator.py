import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.fc = nn.Linear(in_features=5 * 3 * 4, out_features=1)
        self.l_relu = nn.LeakyReLU(0.3)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.l_relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
