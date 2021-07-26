import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class ConvNet(nn.Module):
    def __init__(self, conv_model, fc_tail_lst=[], conv_head_lst=[],
                 flatten=None, softmax_flag=False):
        """
        conv_model:     A convolutional model or a backbone
        fc_tail_lst:    Fully connected layers as list, to create the network's tail
        conv_head_lst:  Convolution layers as list, to create the network's head
        flatten:        A flatten layer if necessary
        softmax_flag:   Determines whether to use softmax on the output or not
        """

        super(ConvNet, self).__init__()

        self.conv_head = None
        if len(conv_head_lst) > 0:
            self.conv_head = nn.Sequential(*conv_head_lst)

        self.conv_model = conv_model

        self.flatten = flatten

        self.fc_tail = None
        if len(fc_tail_lst) > 0:
            self.fc_tail = nn.Sequential(*fc_tail_lst)

        self.softmax_flag = softmax_flag

    def forward(self, x):
        if self.conv_head is not None:
            x = self.conv_head(x)

        x = self.conv_model(x)

        if self.flatten is not None:
            x = F.relu(self.flatten(x))

        if self.fc_tail is not None:
            x = self.fc_tail(x)

        if self.softmax_flag:
            x = F.softmax(x, 1)
        return x


class SiameseResnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, pretrained=False):
        super(SiameseResnet, self).__init__()

        backbone = torchvision.models.resnet50(pretrained=pretrained)
        backbone = torch.nn.Sequential(*(list(backbone.children())[1:-1]))
        flatten = nn.Flatten(1, -1)
        fc_tail_lst = [nn.Linear(in_features=2048, out_features=512, bias=True)]
        conv_head_lst = [nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
        self.convnet = ConvNet(backbone, fc_tail_lst, conv_head_lst, flatten)

    def forward(self, x1, x2):
        out1, out2 = self.convnet(x1), self.convnet(x2)
        dist = F.pairwise_distance(out1, out2)
        return dist