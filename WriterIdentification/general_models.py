import torch
import torchvision
import math
from torch import nn
from torch.nn import functional as F


def weights_init(m):
    if m == None:
        return
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p or 'bias' in p:
                    torch.nn.init.normal_(m.__getattr__(p), 0.0, 0.02)

def weights_init2(m):
    if m == None:
        return
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p or 'bias' in p:
                    torch.nn.init.normal_(m.__getattr__(p), 0.0, 0.02)


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dims, conv_ks_lst, conv_st_lst, pool_ks=0,
                 pool_st=0, conv_padding=0, pool_padding=0, norm='none',
                 activation='relu', pool_type='none', use_bias=True, num_convs=2):
      
        super(Conv2dBlock, self).__init__()
        self.pool = None
        self.conv_pad, self.pool_pad = None, None
        self.norm, self.activation = None, None
        conv_padding = [0]*len(out_dims) if conv_padding == 0 else conv_padding

        # initialize pooling
        if pool_st > 0 and pool_ks > 0:
            if pool_type == 'avg':
                self.pool = nn.AvgPool2d(pool_ks, (pool_st, pool_st), padding=(pool_padding,pool_padding))
            elif pool_type == 'max':
                self.pool = nn.MaxPool2d(pool_ks, (pool_st, pool_st), padding=(pool_padding,pool_padding))
            elif pool_type == 'none':
                self.pool = None
            else:
                assert 0, "Unsupported pooling type: {}".format(pool_pad_type)

        # initialize normalization
        norm_dim = out_dims[-1]
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = nn.AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv_lst = list()
        for dim, ks, st, cp in zip(out_dims, conv_ks_lst, conv_st_lst, conv_padding):
            self.conv_lst.append(nn.Conv2d(in_dim, dim, ks, st, cp, bias=use_bias))
            in_dim = dim
        self.conv_lst = nn.Sequential(*self.conv_lst)
            

    def forward(self, x):
        for conv in self.conv_lst:
            x = conv(x)
        if self.norm:
            x = self.norm(x)       
        if self.activation:
            x = self.activation(x)
        if self.pool:
            x = self.pool(x)

        return x



class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels_lst,
                 block_ks_lst, block_st_lst, pool_ks_lst=[], pool_st_lst=[], conv_pad_lst=[], 
                 pool_pad_lst=[], norm='none', activation='relu',
                 pool_type_lst=[], use_bias=True, num_convs_per_block=[]):

        super(FeaturePyramidNetwork, self).__init__()

        # Fix number of convolution layers per block if necessary
        if len(num_convs_per_block) == 0:
            num_convs_per_block = [2]*len(out_channels_lst)

        # Fix convolution padding inputs if necessary
        if len(conv_pad_lst)==0:
            conv_pad_lst = [0]*len(out_channels_lst)

        # Fix pooling inputs if necessary
        if len(pool_ks_lst)==0 or len(pool_st_lst)==0:
            pool_ks_lst, pool_st_lst, pool_pad_lst = [[0]*len(out_channels_lst)]*3
            pool_type_lst = [['none']*len(out_channels_lst)]
        elif len(pool_pad_lst)==0 or len(pool_type_lst)==0:
            pool_pad_lst = [0]*len(out_channels_lst)

        self.convblock_lst = list()
        for out_channels, conv_ks_lst, conv_st_lst, pool_ks, pool_st, conv_pad, pool_pad, pool_type, convs_num in zip(out_channels_lst,
        block_ks_lst, block_st_lst, pool_ks_lst, pool_st_lst, conv_pad_lst, pool_pad_lst, pool_type_lst, num_convs_per_block):

            self.convblock_lst.append(Conv2dBlock(in_channels, out_channels, conv_ks_lst,
                                      conv_st_lst, pool_ks, pool_st, conv_pad,
                                      pool_pad, norm, activation, pool_type, use_bias, convs_num))
            
            in_channels = out_channels[-1]

        self.convblock_lst = nn.Sequential(*self.convblock_lst)


    def forward(self, x):
        output_features = list()
        for convblock in self.convblock_lst:
            x = convblock(x)
            output_features.append(x)

        return output_features



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



class ConvLstm(nn.Module):
    def __init__(self, in_channels=8, out_channels=2, embed_size=1024,
                 hidden_size=256, num_layers=1, dropout=0, bidirectional=False,
                 pretrained=False):
      
        super(ConvLstm, self).__init__()

        backbone = torchvision.models.resnet50(pretrained=pretrained)
        backbone = torch.nn.Sequential(*(list(backbone.children())[1:-1]))
        flatten = nn.Flatten(1, 3)
        fc_tail_lst = [nn.Linear(in_features=2048, out_features=embed_size, bias=True)]
        conv_head_lst = [nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
        self.conv_encoder = ConvNet(backbone, fc_tail_lst, conv_head_lst, flatten)

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        lstm_out_size = hidden_size*2 if bidirectional else hidden_size

        self.fc = nn.Linear(lstm_out_size, out_channels)

       
    def forward(self, x):
        # Extract h_0, c_0 from the first timestamp
        # x.shape = B x T x 1 x H x W
        T = x.size(1)
        
        x_t = self.conv_encoder(x[:,0,:,:,:])
        output, (h_t, c_t) = self.lstm(x_t.unsqueeze(1))

        for t in range(1, T): # Exclude the first timestamp
            x_t = self.conv_encoder(x[:,t,:,:,:])  
            output, (h_t, c_t) = self.lstm(x_t.unsqueeze(1), (h_t, c_t))

        return self.fc(output.squeeze(1))



class ConvAttentionLstm(nn.Module):
    def __init__(self, in_channels=8, out_channels=256,
                 hidden_size=512, num_layers=1, dropout=0, bidirectional=False,
                 pretrained=False):
      
        super(ConvAttentionLstm, self).__init__()

        # backbone = torchvision.models.resnet50(pretrained=pretrained)
        backbone = torchvision.models.vgg19_bn(pretrained=pretrained)
        backbone = torch.nn.Sequential(*(list(backbone.children())[1:-1]))
        conv_head_lst = [nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
        self.conv_encoder = ConvNet(backbone, conv_head_lst=conv_head_lst)
        self.flatten = nn.Flatten(1, -1)
        self.hidden_size = hidden_size
        self.embed = nn.Linear(in_features=3136, out_features=self.hidden_size, bias=True)

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        lstm_out_size = hidden_size*2 if bidirectional else hidden_size

        self.fc = nn.Linear(lstm_out_size, out_channels)
        

    def attention(self, current_encoder_state, last_state):
        """
        Given Two Tensors: current_encoder_state, last_state with shape of (B, H)
        where B is the batch size and H is the hidden size.
        Returns the output using the attention mechanism
        """

        assert current_encoder_state.shape == last_state.shape and len(current_encoder_state.shape)==2, "Invalid input shapes!"
        B, H = current_encoder_state.shape

        # Reshape input tensors to shape of (B, sqrt(H), sqrt(H))
        current_encoder_state = torch.reshape(current_encoder_state, (B, int(math.sqrt(H)), int(math.sqrt(H))))
        last_state = torch.reshape(last_state, (B, int(math.sqrt(H)), int(math.sqrt(H))))

        # Compute the energy score and the attention weights
        last_state = torch.transpose(last_state, 1,2)
        energy = torch.bmm(current_encoder_state, last_state)
        b, h, w = energy.shape
        attention_weights = torch.reshape(F.softmax(torch.flatten(energy, 1), 1), (b, h, w))

        # Compute the output by a matrix multiplication between the attention weights and the current_encoder_state
        current_encoder_state = torch.transpose(current_encoder_state, 1,2)
        output = torch.bmm(current_encoder_state, attention_weights)
        return output


    def forward(self, x):
        # Extract h_0, c_0 from the first timestamp
        # x.shape = B x T x 1 x H x W
        T = x.size(1)
        
        # Encode first timestamp image into shape of (B, 1, FH, FW)
        # where FW/FH are the dimensions of the features map
        x_t = self.conv_encoder(x[:,0,:,:,:])
        x_t = self.embed(self.flatten(x_t))

        # Extract the first hidden and cell states
        output, (h_t, c_t) = self.lstm(x_t.unsqueeze(1))

        for t in range(1, T): # Exclude the first timestamp

            # Encode the next timestamp image into shape of (B, 1, FH, FW)
            x_t = self.conv_encoder(x[:,t,:,:,:])
            x_t = self.embed(self.flatten(x_t))

            # Attention
            x_t = self.attention(x_t, h_t.squeeze(0))

            # Extract the next hidden and cell states
            output, (h_t, c_t) = self.lstm(self.flatten(x_t).unsqueeze(1), (h_t, c_t))

        return self.fc(output.squeeze(1))


