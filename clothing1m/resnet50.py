import torch.utils.model_zoo as model_zoo
from food101n.meta_layers import *
__all__ = ['ResNet',  'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(MetaModule):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = MetaBatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = MetaBatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.conv3 = MetaConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = MetaBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(MetaModule):

    def __init__(self, block, layers, num_classes=14):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc14 = MetaLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, x.size()[3])
        emb = x.view(x.size(0), -1)
        x = self.fc14(emb)

        return x


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dicts = model.state_dict()
        pretrained_dict = {k:v for k, v in pretained_dict.items() if k in model_dicts}
        model_dicts.update(pretrained_dict)
        model.load_state_dict(model_dicts)
    return model


class baseline_VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(baseline_VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class wpi(MetaModule):
    def __init__(self, input, hidden1, output):
        # 64, 128, 10
        super(wpi, self).__init__()

        self.linear1 = MetaLinear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = MetaLinear(hidden1, hidden1)
        self.linear_mean = MetaLinear(hidden1, output)
        self.linear_var = MetaLinear(hidden1, output)

        self.cls_emb = nn.Embedding(output, 10)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        mean = self.linear_mean(h2)
        log_var = self.linear_var(h2)
        return mean, log_var

    def forward(self, feat, target, sample_num):
        target = self.cls_emb(target)

        x = torch.cat([feat, target], dim=-1)

        mean, log_var = self.encode(x)  # or 100
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std.unsqueeze(0).repeat(sample_num,1,1))
        # print(eps.size())

        return F.sigmoid(mean + std*eps)


class wpi_dec(MetaModule):
    def __init__(self, input, hidden1, output):
        # 64, 128, 10
        super(wpi_dec, self).__init__()

        self.linear1 = MetaLinear(input, hidden1)
        self.tanh = nn.Tanh()
        self.linear2 = MetaLinear(hidden1, hidden1)
        self.linear_mean = MetaLinear(hidden1, output)

        self.cls_emb = nn.Embedding(output, 10)  # or 100

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.linear_mean.weight)
        self.linear_mean.bias.data.zero_()

    def encode(self, x):
        h1 = self.tanh(self.linear1(x))
        h2 = self.tanh(self.linear2(h1))
        mean = self.linear_mean(h2)
        return mean

    def forward(self, feat, target):
        target = self.cls_emb(target)

        x = torch.cat([feat, target], dim=-1)
        mean = self.encode(x) # [100, 10]
        return F.sigmoid(mean)
