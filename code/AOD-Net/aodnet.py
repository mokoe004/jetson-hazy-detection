import torch
import torch.nn as nn

class aodnet(nn.Module):
    def __init__(self):
        super(aodnet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
        self.e_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

        self.e_conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True)
        self.e_conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3, bias=True)

        self.e_conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        #concatenating the output of the two convolutions x1 and x2
        concat1 = torch.cat([x1, x2], 1)
        x3 = self.relu(self.e_conv3(concat1))

        #concatenating the output of the two convolutions x2 and x3
        concat2 = torch.cat([x2, x3], 1)
        x4 = self.relu(self.e_conv4(concat2))

        #concatenating the output of the two convolutions x1, x2, x3 and x4
        concat3 = torch.cat([x1, x2, x3, x4], 1)
        x5 = self.relu(self.e_conv5(concat3))

        #J(x) = T(x)*I(x)-T(x) + B
        clean_image = self.relu((x5*x)-x5+1)

        #return J(x)
        return clean_image
        
