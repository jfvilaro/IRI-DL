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

        self.linear2 = nn.Linear(descriptor_size, 2)


        # TODO this is just an example, this should load pytorch pretrained weights instead random
        self.init_weights(self)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self._in_planes, planes, stride))
            self._in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, data, get_descriptor):
        if get_descriptor == 'True':
            out = data
            out = self.conv1(out)
            out = F.relu(out)
            out = self.conv2(out)
            out = F.relu(out)

            out = out.view(out.shape[0], -1)
            res = self.linear1(out)
            res = self.linear2.weight[0]*res + self.linear2.bias[0]
          #res = self.linear2.weight[1] * res + self.linear2.bias[1]

            # metrica per considerar que són iguals weight[0]
            # metrica per considerar que considerar que són diferents weight[1]

        else:
            res = []
            for i in range(2):  # Siamese nets; sharing weights
                out = data[i]
                out = self.conv1(out)
                out = F.relu(out)
                out = self.conv2(out)
                out = F.relu(out)

                out = out.view(out.shape[0], -1)
                out = self.linear1(out)
                res.append(F.relu(out))

            res = torch.abs(res[1] - res[0])
            res = self.linear2(res)

        return res

    def descriptor(self, data):

        out = data
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)

        out = out.view(out.shape[0], -1)
        learned_feature = self.linear1(out)

        return learned_feature
