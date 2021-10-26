import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.bn2(self.conv2(out))
        # print(out.shape)
        out += self.shortcut(x)
        # print(out.shape)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_plane, block, num_blocks, hidden_dim=512, out_dim=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_plane, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(512 * block.expansion, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.img_output_dim = None
        self.drop_path_prob = 0.0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            # print(nn.Sequential(*layers))
        return nn.Sequential(*layers)

    def extract_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        return out

    def sub_forward(self, x):
        x = self.extract_feature(x)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        # print(x.shape)
        return x

    def forward(self, x0, x1):
        x0 = self.sub_forward(x0)

        if self.img_output_dim is None:
            self.img_output_dim = x0.shape[1]

        x1 = self.sub_forward(x1)
        diff = torch.abs(x0 - x1)
        scores = self.fc2(diff)
        scores = torch.reshape(scores, (-1,))
        # print(scores.shape)
        return scores


class MLP_classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=10):
        super(MLP_classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = x.detach()
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out



def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34(in_plane):
    return ResNet(in_plane, BasicBlock, [3, 4, 6, 3])


def ResNet50(in_plane):
    return ResNet(in_plane, Bottleneck, [3, 4, 6, 3])


def ResNet101(in_plane):
    return ResNet(in_plane, Bottleneck, [3, 4, 23, 3])


def ResNet152(in_plane):
    return ResNet(in_plane, Bottleneck, [3, 8, 36, 3])

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    x0 = torch.rand(128, 1, 64, 64).to(device)
    net = ResNet34(1).to(device)
    out = net(x0, x0)
    print(out)