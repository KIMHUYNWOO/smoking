import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np

def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet, self).__init__()
        seed_everything(42)
        
        self.in_planes = 2

        self.conv1 = nn.Conv2d(3, 2, kernel_size=3,
                               stride=1, padding=1, bias=False) #32 x 2 x 224 x 224
        self.bn1 = nn.BatchNorm2d(2)
        self.layer1 = self._make_layer(2, 2, stride=1) #32 x 2 x 224 x 224
        self.layer2 = self._make_layer(4, 2, stride=1) #32 x 4 x 224 x 224
        self.layer3 = self._make_layer(8, 2, stride=1) #32 x 8 x 224 x 224
        self.layer4 = self._make_layer(12, 2, stride=1) #32 x 12 x 224 x 224
        self.layer5 = self._make_layer(16, 2, stride=1) #32 x 16 x 224 x 224
        self.layer6 = self._make_layer(24, 2, stride=1) #32 x 24 x 224 x 224
        self.layer7 = self._make_layer(32, 2, stride=1) #32 x 32 x 224 x 224
        self.layer8 = self._make_layer(32, 2, stride=2) #32 x 32 x 112 x 112
        self.layer9 = self._make_layer(64, 2, stride=2) #32 x 64 x 56 x 66 
        self.layer10 = self._make_layer(128, 2, stride=2) #32 x 128 x 56 x 56
        self.layer11 = self._make_layer(128, 2, stride=2) #32 x 128 x 28 x 28
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(6272, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) #32 x 2 x 224 x 224
        out = self.layer1(out) #32 x 2 x 224 x 224
        out = self.layer2(out) #32 x 4 x 224 x 224
        out = self.layer3(out) #32 x 8 x 224 x 224
        out = self.layer4(out) #32 x 12 x 224 x 224
        out = self.layer5(out) #32 x 16 x 224 x 224
        out = self.layer6(out) #32 x 24 x 224 x 224
        out = self.layer7(out) #32 x 32 x 224 x 224
        out = self.layer8(out) #32 x 32 x 112 x 112
        out = self.layer9(out) #32 x 64 x 56 x 56 
        out = self.layer10(out) #32 x 128 x 28 x 28
        out = self.layer11(out) #32 x 128 x 14 x 14
        out = F.avg_pool2d(out, 2) #batch_size x 128 x 7 x 7
        out = self.flatten(out) # batch_size x 6272
        out = self.linear(out) #batch_size x 1
        out = self.sigmoid(out) #batch_size x 1
        return out