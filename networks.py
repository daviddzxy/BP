from torch import nn


class SiameseNet(nn.Module):
    def __init__(self, network):
        super(SiameseNet, self).__init__()
        self.network = network

    def forward(self, input1, input2):
        output1 = self.network(input1)
        output2 = self.network(input2)
        return output1, output2

    def get_encoding(self, input):
        return self.network(input)


class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()

    def forward(self, _input):
        output = self.cnn1(_input)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output


class Net3DChannel1(ConvNetwork):
    def __init__(self):
        super(Net3DChannel1, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(4),

            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),

            nn.Conv3d(8, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(12),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(12 * 5 * 24 * 24, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )


class Net2DChannel1(ConvNetwork):
    def __init__(self):
        super(Net2DChannel1, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(12 * 24 * 24, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )


class Net2DChannel2(ConvNetwork):
    def __init__(self):
        super(Net2DChannel2, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2)
        )





