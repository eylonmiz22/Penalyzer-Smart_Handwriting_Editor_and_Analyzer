import torch
import torchvision
import math
from torch import nn
from torch.nn import functional as F

from general_models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FragmentPath(nn.Module):
    def __init__(self, in_channels_lst, out_channels_lst,
                 block_ks_lst, block_st_lst, pool_ks_lst=[], pool_st_lst=[], conv_pad_lst=[], 
                 pool_pad_lst=[], norm='none', activation='relu', 
                 conv_pad_type_lst=[], pool_pad_type_lst=[],
                 pool_type_lst=[], use_bias=True, num_convs_per_block=[]):

        super(FragmentPath, self).__init__()

        # Fix number of convolution layers per block if necessary
        if len(num_convs_per_block) == 0:
            num_convs_per_block = [2]*len(out_channels_lst)

        # Fix convolution padding inputs if necessary
        if len(conv_pad_lst)==0:
            conv_pad_lst = [0]*len(out_channels_lst)

        # Fix pooling inputs if necessary
        if len(pool_ks_lst)==0 or len(pool_st_lst)==0:
            pool_ks_lst, pool_st_lst, pool_pad_lst = [[0]*len(out_channels_lst)]*2
            pool_type_lst = [['none']*len(out_channels_lst)]
        elif len(pool_pad_lst)==0 or len(pool_type_lst)==0:
            pool_pad_lst = [0]*len(out_channels_lst)

        self.convblock_lst = list()
        i, count = 0, len(in_channels_lst)
        for in_channels, out_channels, conv_ks_lst, conv_st_lst, pool_ks, pool_st, conv_pad, pool_pad, pool_type, convs_num in zip(in_channels_lst, out_channels_lst,
        block_ks_lst, block_st_lst, pool_ks_lst, pool_st_lst, conv_pad_lst, pool_pad_lst, pool_type_lst, num_convs_per_block):
            if i+1 == count:
                activation = 'none'
            self.convblock_lst.append(Conv2dBlock(in_channels, out_channels, conv_ks_lst,
                                      conv_st_lst, pool_ks, pool_st, conv_pad,
                                      pool_pad, norm, activation, pool_type, use_bias, convs_num))
            i += 1
            
        self.convblock_lst = nn.Sequential(*self.convblock_lst)


    def forward(self, fragments_t, fpn_fragment_t):
        """
        fragments_t: Fragments in time_t (=step_t) of all batch word images.
        fragments_t.shape is (B, C, H, WS) Where C = 1 (Grayscale) and WS is the
        width timestep int(W/self.timestamps)

        fpn_fragment_t: The cropped FPN (batch) features by timestamp t, given as a list.
        The elements in the list have different shapes (from the high-level 
        features to the low_level ones)
        """

        for convblock, f_t in zip(self.convblock_lst, fpn_fragment_t):
            B1, C1, H1, W1 = fragments_t.shape
            B2, C2, H2, W2 = f_t.shape
            assert B1 == B2, "Incompatible batch size"

            # Resize f_t to match the current fragments_t shape
            # Note that it is not recommended to resize tensors drastically
            # at the middle of the forward process, to keep the quality of the data.
            # Hence the resize should be gentle.
            f_t = F.interpolate(f_t, (H1, W1))
            
            # Concatenate the fpn cropped features and fragments_t
            # by the channel's dimension
            fragments_t = torch.cat([fragments_t, f_t], dim=1)
            fragments_t = convblock(fragments_t)
            
        return fragments_t



class FragNet(nn.Module):
    def __init__(self, fpn_in_channels, num_writers, num_features=2048*3*3,
                 fpn_out_channels=[[48, 64], [96, 128], [196, 256], [384, 512]],
                 fragments_out_channels=[[48, 64], [196, 256], [448, 512], [896, 1024], [1790, 2048]],
                 fragments_in_channels=[1+1, 64+64, 128+256, 256+512, 512+1024],
                 q=64, height_thresh=64//8, width_thresh=64//2, 
                 pixel_thresh=0, device="cpu"):
    
        super(FragNet, self).__init__()

        self.q = q
        self.height_thresh = height_thresh
        self.width_thresh = width_thresh
        self.pixel_thresh = pixel_thresh
        self.device = device

        self.fpn_pathway = FeaturePyramidNetwork(fpn_in_channels, fpn_out_channels,
                                                [[3,3], [3,3], [3,3], [3,3]], [[1,1]]*4,
                                                [0,2,2,2], [0,2,2,2], norm='bn',
                                                conv_pad_lst=[[3//2]*2]*4, pool_pad_lst=[2//2]*4,
                                                pool_type_lst=['none','max','max','max'])

        self.fragment_pathway = FragmentPath(fragments_in_channels, fragments_out_channels,
                                             [[3,3], [3,3], [3,3], [3,3], [3,3]], [[1,1]]*5,
                                             [0,3,3,3,3], [0,2,2,2,2], norm='bn',
                                             conv_pad_lst=[[1,1],[1,1],[1,1],[1,1],[1,1]],
                                             pool_pad_lst=[0]*5, pool_type_lst=['none','max','max','max','avg'])
        
        self.flatten = nn.Flatten(1, -1)
        self.classifier = nn.Linear(num_features, num_writers)


    def forward(self, words):
        """
        words:     The input batch of words with shape of (B, C, H, W)
        """

        # Extract the different features from the given words
        word_features = self.fpn_pathway(words)
        words_batch, words_channels, words_height, words_width = words.shape

        # The loop ends in (length/q - 1) because the last step is 
        # words[:, :, (length-1)*q:length*q, (length-1)*q:length*q]
        outputs = list()
        for i in range(0, words_height//self.q):
            for j in range(0, words_width//self.q):
            
                # Get fragments q-ij from the word images
                batch_fragments_t = words[:,:,i*self.q:(i+1)*self.q,j*self.q:(j+1)*self.q]
                
                fragments_batch, fragments_channels, fragments_height, fragments_width = batch_fragments_t.shape 
                if fragments_height < self.height_thresh or fragments_width < self.width_thresh or torch.sum(batch_fragments_t) < self.pixel_thresh:
                    continue

                # Get fragments in time t from the FPN outputs
                # The first fp fragment equals the fragment itself
                batch_wf_fragment_t = [batch_fragments_t]

                for wf in word_features:
                    wf_batch, wf_channels, wf_height, wf_width = wf.shape
                    wf_height_q = int(self.q*(wf_height/words_height))
                    wf_width_q = int(self.q*(wf_width/words_width))
                    batch_wf_fragment_t.append(wf[:,:,i*wf_height_q:(i+1)*wf_height_q,j*wf_width_q:(j+1)*wf_width_q])
                
                # Pass the input fragments to the fragment pathway
                output = self.fragment_pathway(batch_fragments_t, batch_wf_fragment_t)
                output = self.classifier(self.flatten(output))
                outputs.append(output)

        return outputs



class SimpleFragNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, out_word_features=256,
                 out_fragments_features=256, embed_size=1024, hidden_size=1024,
                 num_layers=1, dropout=0, num_timestamps=4, bidirectional=False,
                 pretrained=False):
      
        super(SimpleFragNet, self).__init__()

        backbone = torchvision.models.resnet50(pretrained=pretrained)
        backbone = torch.nn.Sequential(*(list(backbone.children())[1:-1]))
        flatten = nn.Flatten(1, -1)
        fc_tail_lst = [nn.Linear(in_features=2048, out_features=1024, bias=True),
                      nn.Linear(in_features=1024, out_features=out_word_features, bias=True)]
        conv_head_lst = [nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
        self.word_model = ConvNet(backbone, fc_tail_lst, conv_head_lst, flatten)
        
        self.fragments_model = ConvAttentionLstm(in_channels, out_fragments_features, hidden_size, num_layers, dropout, bidirectional, pretrained)
        # self.fragments_model = ConvLstm(in_channels, out_fragments_features, embed_size, 512, num_layers, dropout, bidirectional, pretrained)
        self.timestamps = num_timestamps

        self.fc = nn.Linear(out_word_features+out_fragments_features, out_channels)

       
    def forward(self, words):
        global device

        B, _, H, W = words.shape
        step = W // self.timestamps
        last_width_offset = W - (W // self.timestamps)        
        fragments = torch.zeros(B, self.timestamps, 1, H, step, requires_grad=True).to(device)

        for b in range(B):
            for t, width_offset in zip(range(self.timestamps), range(0, last_width_offset, step)):
                fragments[b,t,:,:,:] = words[b,:,:,width_offset:width_offset+step]

        fragments_features = self.fragments_model(fragments)
        words_features = self.word_model(words)

        x = torch.cat([words_features, fragments_features], dim=1).requires_grad_()
        x = F.relu(x)
        return self.fc(x)



class AttentionFragNet(nn.Module):
    def __init__(self, fpn_in_channels, num_writers, num_features=2048*3*3,
                 fpn_out_channels=[[48, 64], [96, 128], [196, 256], [384, 512]],
                 fragments_out_channels=[[48, 64], [196, 256], [448, 512], [896, 1024], [1790, 2048]],
                 fragments_in_channels=[1+1, 64+64, 128+256, 256+512, 512+1024], 
                 lstm_num_layers=1, embed_size=1024, hidden_size=1024, dropout=0,
                 bidirectional=False, num_layers=1, timestamps=4, width_thresh=(216//4)//2,
                 pixel_thresh=0, device="cpu"):
    
        super(AttentionFragNet, self).__init__()

        self.first_fpn_output_shape = fpn_out_channels[0][-1]
        self.timestamps = timestamps
        self.width_thresh = width_thresh
        self.pixel_thresh = pixel_thresh
        self.device = device

        self.fpn_pathway = FeaturePyramidNetwork(fpn_in_channels, fpn_out_channels,
                                                [[3,3], [3,3], [3,3], [3,3]], [[1,1]]*4,
                                                [0,2,2,2], [0,2,2,2], norm='bn',
                                                conv_pad_lst=[[3//2]*2]*4, pool_pad_lst=[2//2]*4,
                                                pool_type_lst=['none','max','max','max'])

        self.fragment_pathway = FragmentPath(fragments_in_channels, fragments_out_channels,
                                             [[3,3], [3,3], [3,3], [3,3], [3,3]], [[1,1]]*5,
                                             [0,3,3,3,3], [0,2,2,2,2], norm='bn',
                                             conv_pad_lst=[[1,1],[1,1],[1,1],[1,1],[1,1]],
                                             pool_pad_lst=[0]*5, pool_type_lst=['none','max','max','max','avg'])
        
        self.flatten = nn.Flatten(1, -1)
        self.embed = nn.Linear(num_features, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, lstm_num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)

        lstm_out_size = hidden_size*2 if bidirectional else hidden_size
        self.classifier = nn.Linear(lstm_out_size, num_writers) if bidirectional else nn.Linear(hidden_size, num_writers)


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


    def forward(self, words, h_t=None, c_t=None):
        """
        words:     The input batch of words with shape of (B, C, H, W)
        h_t, c_t:  The decoder states
        """

        # Extract the different features from the given words
        word_features = self.fpn_pathway(words)

        B1, C1, H1, W1 = words.shape
        step = W1 // self.timestamps

        # The loop ends in self.timestamps-1 because the last timestamp is 
        # words[:,:,:,(length-1)*step:length*step]
        for t in range(self.timestamps-1):
            
            # Get fragments in time t from the word images
            batch_fragments_t = words[:,:,:,t*step:(t+1)*step]
            fragments_batch, fragments_channels, fragments_height, fragments_width = batch_fragments_t.shape 
            if fragments_width < self.width_thresh or torch.sum(batch_fragments_t) < self.pixel_thresh:
                continue

            # Get fragments in time t from the FPN outputs
            # The first fp fragment equals the fragment itself
            batch_fp_fragment_t = [batch_fragments_t]
            for wf in word_features:
                B2, C2, H2, W2 = wf.shape
                fpn_step = W2 // self.timestamps
                batch_fp_fragment_t.append(wf[:,:,:,t*fpn_step:(t+1)*fpn_step])

            # Pass the inputs t to the fragment pathway
            output = self.fragment_pathway(batch_fragments_t, batch_fp_fragment_t)

            # Embed the output in the current timestamp and pass it as an input vector to the LSTM layer 
            embedded_t = self.embed(self.flatten(output))

            if t == 0: # First timestamp
                output, (h_t, c_t) = self.lstm(embedded_t.unsqueeze(1))
            elif t > 0: # Attention
                output = self.flatten(self.attention(embedded_t, h_t.squeeze(0)))
                output, (h_t, c_t) = self.lstm(output.unsqueeze(1), (h_t, c_t))  

        output = self.classifier(output.squeeze(1))
        return output


    
class SiameseResnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, pretrained=False):
      
        super(SiameseResnet, self).__init__()

        backbone = torchvision.models.resnet50(pretrained=pretrained)
        backbone = torch.nn.Sequential(*(list(backbone.children())[1:-1]))
        flatten = nn.Flatten(1, -1)
        fc_tail_lst = [nn.Linear(in_features=2048, out_features=512, bias=True)]
        conv_head_lst = [nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
        self.convnet = ConvNet(backbone, fc_tail_lst, conv_head_lst, flatten)
        # self.classifier = nn.Linear(in_features=512, out_features=out_channels, bias=True)
        
    def forward(self, x1, x2):
        out1, out2 = self.convnet(x1), self.convnet(x2)
        dist = F.pairwise_distance(out1, out2)
        return dist



class SigNet(nn.Module):

    """
    The architecture of the paper SigNet: 
    https://arxiv.org/abs/1707.02131
    """

    def __init__(self):
        super(SigNet, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        self.fc = nn.Sequential(
            nn.Linear(30720, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128,2))
        

    def forward_once(self, x):
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


    def forward(self, x1, x2):
        out1, out2 = self.forward_once(x1), self.forward_once(x2)
        dist = F.pairwise_distance(out1, out2)
        # dist = torch.sigmoid(dist)
        return dist


def siamese_factory(model_idx):
    if model_idx == 0:
        return SiameseResnet()
    elif model_idx == 1:
        return SigNet()
    return None


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, batch_size=64):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.batch_size = batch_size

    def forward(self, dist, y):
        dist_sq = torch.pow(dist, 2)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / self.batch_size
        return loss