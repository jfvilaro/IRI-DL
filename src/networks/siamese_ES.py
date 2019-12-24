from src.networks.networks import NetworkBase
import torch.nn as nn
import torch.nn.functional as F
import torch

class SiameseNet(NetworkBase):
    def __init__(self, num_classes=10, descriptor_size=64):
        super(SiameseNet, self).__init__()
        self._name = "SiameseNet"
        self._in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, 5)
        #self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        #self.conv3 = nn.Conv2d(128, 256, 3)

        self.linear1 = nn.Linear(1152, descriptor_size)

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
