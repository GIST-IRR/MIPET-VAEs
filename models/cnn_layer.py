import torch.nn as nn

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=4, stride=2)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.relu(output)
        return output

class CNNTrasnposedLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNTrasnposedLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.relu(output)
        return output