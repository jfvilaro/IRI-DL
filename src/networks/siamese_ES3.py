from src.networks.networks import NetworkBase
import torch.nn as nn
import torch.nn.functional as F
import torch

class SiameseNet(NetworkBase):
    def __init__(self, num_classes=2, descriptor_size=64, image_size=9):
        super(SiameseNet, self).__init__()
        self._name = "SiameseNet"
        self._in_planes = 64

        kernel_size_1 = 5
        kernel_size_2 = 3
        output_size = image_size - (kernel_size_1-1) - (kernel_size_2-1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size_1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size_2)

        self.linear1 = nn.Linear(output_size*output_size*128, descriptor_size)

        # TODO this is just an example, this should load pytorch pretrained weights instead random
        self.init_weights(self)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self._in_planes, planes, stride))
            self._in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, data):

        out = data
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)

        out = out.view(out.shape[0], -1)
        res = self.linear1(out)

        return res
