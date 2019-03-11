from torch import nn


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()

    def forward_once(self, x):
        pass

    def forward(self, input1, input2):
        pass


class Channel1SiameseNet(SiameseNet):
    def __init__(self):
        super(Channel1SiameseNet, self).__init__()
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

            nn.ReflectionPad2d(1),
            nn.Conv2d(12, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(12),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(12 * 16 * 16, 500),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(500),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(500),

            nn.Linear(500, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class Channel2SiameseNet(SiameseNet):
    def __init__(self):
        super(Channel2SiameseNet, self).__init__()
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

            nn.Linear(512, 2))


    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

