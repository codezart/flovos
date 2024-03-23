from __future__ import division

import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

from helpers import *


class ResBlock(nn.Module):
    """
    A basic residual block that applies a sequence of convolutions,
    followed by a skip connection that adds the input directly to the result of the convolutional layers.
    This helps in mitigating the vanishing gradient problem and allows for deeper networks
    by enabling the flow of gradients through the skip connections.
    """

    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(
                indim, outdim, kernel_size=3, padding=1, stride=stride
            )

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x, inplace=True))
        r = self.conv2(F.relu(r, inplace=True))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


def pixelshuffle_invert(x, factor_hw):
    """
    A function that performs the inverse operation of pixel shuffle,
    which is commonly used for super-resolution tasks.
    It essentially rearranges elements in a tensor from a high-resolution
    feature space to a low-resolution feature space by decreasing height and width
    while increasing the number of channels
    """
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC * (pH * pW), iH // pH, iW // pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)  # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y


class LAE(nn.Module):
    """
    This class seems to combine features from the input mask and object with
    intermediate representations of the input frame at different resolutions (r4, r3, r2, c1)
    using convolutional layers. It likely aims to refine the
    feature representation by incorporating attention to local regions,
    focusing on relevant features for segmentation.
    """

    def __init__(self):
        super(LAE, self).__init__()
        self.conv1_m = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # mask
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2_m = nn.Conv2d(
            256, 1024, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2_o = nn.Conv2d(
            256, 1024, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.conv_fusion1 = nn.Conv2d(
            64, 256, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv_fusion2 = nn.Conv2d(
            256, 512, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv_fusion3 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv_fusion4 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv_fusion5 = nn.Conv2d(
            1024, 1024, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, r4, r3, r2, c1, in_m, in_o):
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float()  # add channel dim
        m_small = pixelshuffle_invert(m, (16, 16))
        o_small = pixelshuffle_invert(o, (16, 16))

        x = F.relu(c1 + self.conv1_m(m) + self.conv1_o(o), inplace=True)
        x = F.relu(r2 + self.conv_fusion1(x), inplace=True)
        x = F.relu(r3 + self.conv_fusion2(x), inplace=True)
        x = F.relu(
            r4 + self.conv2_m(m_small) + self.conv2_o(o_small) + self.conv_fusion3(x),
            inplace=True,
        )
        x = F.relu(self.conv_fusion4(x), inplace=True)
        x = F.relu(self.conv_fusion5(x), inplace=True)
        return x, None, None, None, None


class Encoder(nn.Module):
    """
    Utilizes a pretrained ResNet-50 model to encode input frames into a
    hierarchical set of feature maps at different resolutions (r4, r3, r2, c1).
    These feature maps are later used for extracting key-value pairs and
    for decoding to produce segmentation masks.
    The encoder standardizes inputs by subtracting the mean and dividing by the standard deviation,
    using predefined values that match those used in ImageNet training.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = resnet
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 64
        self.res3 = resnet.layer2  # 1/8, 128
        self.res4 = resnet.layer3  # 1/8, 256

        self.register_buffer(
            "mean", torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std  # normalized frame
        # print("Encoder-forward After normalization:", torch.isnan(f).any())
        # print(torch.isnan(self.conv1.weight).any())  # Check for NaN in weights
        # print(torch.isnan(self.conv1.bias).any())    # Check for NaN in bias if bias=True

        x = self.conv1(f)
        # print(torch.isnan(x).any())    # Check for NaN in bias if bias=True
        # print("x:  {}".format(x))
        c1 = self.bn1(x)
        x = self.relu(c1)  # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    """
    A module designed to refine segmentation masks at different scales.
    It takes feature maps from the encoder and previously generated masks
    to refine the segmentation details through convolutional and residual blocks,
    followed by upsampling.
    """

    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(
            inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1
        )
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(
            pm, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    """
    Decodes the feature maps from the Encoder and the refined features
    from the Refine module to produce the final segmentation mask.
    It includes convolutional layers and a final prediction layer
    that outputs the segmentation logits, which are then upscaled to the original resolution.
    """

    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(512, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)
        self.RF2 = Refine(256, mdim)

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)

        p2 = self.pred2(F.relu(m2, inplace=True))

        p = F.interpolate(p2, scale_factor=4, mode="bilinear", align_corners=False)
        return p


class Memory(nn.Module):
    """
    Implements a memory mechanism for video object segmentation,
    allowing the network to utilize information from previous frames.
    It matches query and key feature maps to generate a weighted sum of value feature maps,
    which is a way of incorporating temporal information into the segmentation process.
    """

    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, N = m_in.size()
        _, D_o, H, W = q_out.size()

        mi = m_in.view(B, D_e, N)
        mi = torch.transpose(mi, 1, 2)

        qi = q_in.view(B, D_e, H * W)

        p = torch.bmm(mi, qi)
        p = p / math.sqrt(D_e)
        p = torch.clamp(p, max=80.0)
        p = torch.exp(p)
        p = p / torch.sum(p, dim=1, keepdim=True)

        mo = m_out.view(B, D_o, N)
        mem = torch.bmm(mo, p)
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out


class KeyValue(nn.Module):
    """
    A module that generates key and value pairs from feature maps.
    These pairs are used in the Memory module to link features across frames,
    enabling the network to understand changes in object appearance and motion over time.
    """

    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(
            indim, keydim, kernel_size=(3, 3), padding=(1, 1), stride=1
        )
        self.Value = nn.Conv2d(
            indim, valdim, kernel_size=(3, 3), padding=(1, 1), stride=1
        )

    def forward(self, x):
        return self.Key(x), self.Value(x)


class _ASPPModule(nn.Module):
    """
    Implements the ASPP architecture, which applies atrous convolution with different
    dilation rates to capture multi-scale information.
    This is particularly useful for segmentation tasks where objects can appear at various sizes.
    """

    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x):
        x = self.atrous_conv(x)
        return F.relu(x, inplace=True)


class ASPP(nn.Module):
    """
    ASPP is effective in enlarging the receptive field and capturing contextual information at multiple scales.
    """

    def __init__(self):
        super(ASPP, self).__init__()
        dilations = [1, 2, 4, 8]

        self.aspp1 = _ASPPModule(256, 128, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(
            256, 128, 3, padding=dilations[1], dilation=dilations[1]
        )
        self.aspp3 = _ASPPModule(
            256, 128, 3, padding=dilations[2], dilation=dilations[2]
        )
        self.aspp4 = _ASPPModule(
            256, 128, 3, padding=dilations[3], dilation=dilations[3]
        )
        self.conv1 = nn.Conv2d(512, 512, 1, bias=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        return F.dropout(F.relu(x, inplace=True), p=0.5, training=self.training)


class SwiftNet(nn.Module):
    """
    The main network that integrates all the components mentioned above.
    It defines a process for memorizing key-value pairs from previous frames
    and using them to segment objects in current frames.
    The network is capable of adapting to both the initial frame of a video, where it memorizes the scene,
    and subsequent frames, where it uses the memory to assist in segmentation.
    """

    def __init__(self):
        super(SwiftNet, self).__init__()
        self.LAE = LAE()
        self.Encoder = Encoder()

        self.KV_M_r4 = KeyValue(1024, keydim=32, valdim=128)
        self.KV_Q_r4 = KeyValue(1024, keydim=32, valdim=128)

        self.Memory = Memory()
        self.Decoder = Decoder(128)
        self.aspp = ASPP()

    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(
                torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3])
            )
            pad_mem[0, 1 : num_objects + 1, :, 0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def Memory_update(self, keys, values, prev_key, prev_value):
        beta = 0.1
        n, _, _ = keys.shape
        keys_norm = keys / torch.norm(keys, p=2, dim=1, keepdim=True)
        prev_key_norm = prev_key / torch.norm(prev_key, p=2, dim=1, keepdim=True)
        prev_key_norm = prev_key_norm.permute(0, 2, 1)
        coor = torch.bmm(prev_key_norm, keys_norm)
        sim, _ = torch.max(coor, dim=2)
        sim_, mid = torch.max(coor, dim=1)
        _, ind = torch.sort(sim, dim=1, descending=True)
        _, ind = torch.sort(ind, dim=1)
        ind = ind.unsqueeze(1)
        N = ind.shape[2]
        prev_key_slt = prev_key[ind.expand(-1, 32, -1) >= int(N * (1 - beta))].reshape(
            n, 32, -1
        )
        prev_value_slt = prev_value[
            ind.expand(-1, 128, -1) >= int(N * (1 - beta))
        ].reshape(n, 128, -1)
        return torch.cat([keys, prev_key_slt], dim=2), torch.cat(
            [values, prev_value_slt], dim=2
        )

    def Soft_aggregation(self, ps, K):
        # stm
        # num_objects, H, W = ps.shape
        # em = ToCuda(torch.zeros(1, K, H, W))
        # em[0, 0] = torch.prod(1 - ps, dim=0)  # bg prob
        # em[0, 1 : num_objects + 1] = ps  # obj prob
        # em = torch.clamp(em, 1e-7, 1 - 1e-7)
        # logit = torch.log((em / (1 - em)))

        # swiftnet
        num_objects, H, W = ps.shape
        bg_prob = torch.prod(1 - ps, dim=0).unsqueeze(0).unsqueeze(0)  # bg prob
        em = torch.cat([bg_prob, ps.unsqueeze(0)], dim=1)
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def memorize(
        self, frame, masks, r4, r3, r2, c1, num_objects, first_frame_flag=False
    ):
        num_objects = num_objects[0].item()
        _, K, H, W = masks.shape

        (frame, masks), pad = pad_divide_by(
            [frame, masks], 64, (frame.size()[2], frame.size()[3])
        )

        B_list = {"f": [], "m": [], "o": []}
        for o in range(1, num_objects + 1):
            B_list["f"].append(frame)
            B_list["m"].append(masks[:, o])
            B_list["o"].append(
                (
                    torch.sum(masks[:, 1:o], dim=1)
                    + torch.sum(masks[:, o + 1 : num_objects + 1], dim=1)
                ).clamp(0, 1)
            )

        B_ = {}
        for arg in B_list.keys():
            B_[arg] = torch.cat(B_list[arg], dim=0)

        if first_frame_flag == True:
            logging.info("First frame flag is set to True")
            r4, r3, r2, c1, _ = self.Encoder(B_["f"])  # r4, r3, r2, c1, f
            logging.info(
                "Encoder for FFF. r4: \nshape{}, \nr3: \nshape{}, \nr2:\nshape{}, \nc1:\nshape{}, \n".format(
                    r4.size(), r3.size(), r2.size(), c1.size()
                )
            )

        logging.info("Entering LAE ")
        r4, _, _, _, _ = self.LAE(r4, r3, r2, c1, B_["m"], B_["o"])
        logging.info("LAE out r4: \nshape: {}".format(r4.size()))

        k4, v4 = self.KV_M_r4(r4)
        logging.info(
            "KVMr4 k4: \nshape: {}\nv4: \nshape: {}".format(k4.size(), v4.size())
        )
        k4 = k4.unsqueeze(0).unsqueeze(3)
        v4 = v4.unsqueeze(0).unsqueeze(3)
        k4 = k4.reshape(num_objects, 32, -1)
        v4 = v4.reshape(num_objects, 128, -1)
        logging.info(
            "After unsqueeze and reshape k4: \nshape: {}\nv4: \nshape: {}".format(
                k4.size(), v4.size()
            )
        )

        return k4, v4

    def segment(self, frame, keys, values, num_objects):
        num_objects = num_objects[0].item()
        # _, keydim, N = keys.shape
        [frame], pad = pad_divide_by([frame], 64, (frame.size()[2], frame.size()[3]))
        r4, r3, r2, c1, _ = self.Encoder(frame)
        k4, v4 = self.KV_Q_r4(r4)
        k4e, v4e = k4.expand(num_objects, -1, -1, -1), v4.expand(
            num_objects, -1, -1, -1
        )
        r3e, r2e = r3.expand(num_objects, -1, -1, -1), r2.expand(
            num_objects, -1, -1, -1
        )
        m4 = self.Memory(keys, values, k4e, v4e)
        m4 = self.aspp(m4)
        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:, 1]
        logit = self.Soft_aggregation(ps, 11)
        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2] : -pad[3], :]
        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0] : -pad[1]]
        return logit, r4, r3, r2, c1

    def forward(self, *args, **kwargs):
        logging.info("SwiftNet: forward pass {}".format(len(args) + len(kwargs)))
        if len(args) + len(kwargs) <= 4:
            logging.info("SwiftNet-forward: entering segment function")
            return self.segment(*args, **kwargs)
        else:
            logging.info("SwiftNet-forward: entering memorize function")
            return self.memorize(*args, **kwargs)
