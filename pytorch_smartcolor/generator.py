import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(in_features=100, out_features=8 * 5 * 3, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=8 * 5 * 3)
        self.l_relu = nn.LeakyReLU(0.3)
        self.drop = nn.Dropout(0.5)
        self.convT2d1 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), bias=False,
                                           padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=4)

        self.convT2d2 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=(1, 1), bias=False,
                                           padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn1(x)
        x = self.l_relu(x)
        x = x.view(-1, 8, 5, 3)
        x = self.drop(x)
        x = self.convT2d1(x)
        x = self.bn2(x)
        x = self.l_relu(x)
        x = self.convT2d2(x)
        x = self.drop(x)
        x = self.tanh(x)
        return x
