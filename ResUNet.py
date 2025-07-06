import os
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

import torch.cuda.amp as amp
import matplotlib
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

import warnings
warnings.filterwarnings("ignore")

#channel attention block
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
        
#spatial attention block
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_out)
        return self.sigmoid(x_out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

       
        self.ca = ChannelAttention(skip_channels)
        self.sa = SpatialAttention()

    def forward(self, x, skip):
        x = self.up(x)

       
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

       
        skip = skip * self.ca(skip)
        skip = skip * self.sa(skip)

       
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResUNet(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=True):
        super(ResUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Load ResNet34 backbone
        resnet = resnet34(weights=None)
        state_dict = torch.load('./resnet34.pth')  
        resnet.load_state_dict(state_dict)

        # Encoder
        self.firstconv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)

        # Segmentation heads
        self.dsv4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.dsv3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.dsv2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.dsv1 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Classification head 
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 5),  # 5 classes
            nn.Softmax(dim=1)
        )

        # Router 
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
      
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_skip1 = x  # 64ch, 1/2
        x = self.firstmaxpool(x)

        x = self.encoder1(x)
        x_skip2 = x  # 64ch, 1/4

        x = self.encoder2(x)
        x_skip3 = x  # 128 ch, 1/8

        x = self.encoder3(x)
        x_skip4 = x  # 256 ch, 1/16

        x4 = self.encoder4(x)  # 512ch, 1/32
        
        # routing weights
        routing_weights = self.router(x4)
        topk_val, topk_idx = torch.topk(routing_weights, k=1, dim=-1)  # [B, 1]
        
        x = self.decoder4(x4, x_skip4)
        d4 = x  

        x = self.decoder3(x, x_skip3)
        d3 = x 

        x = self.decoder2(x, x_skip2)
        d2 = x  

        x = self.decoder1(x, x_skip1)
        d1 = x  

        # Final segmentation outputs
        final = self.final(x)
        final = F.interpolate(final, size=input_size, mode='bilinear', align_corners=True)
        
        # Multi-scale outputs
        p4 = self.dsv4(d4)
        p4 = F.interpolate(p4, size=input_size, mode='bilinear', align_corners=True)
        
        p3 = self.dsv3(d3)
        p3 = F.interpolate(p3, size=input_size, mode='bilinear', align_corners=True)
        
        p2 = self.dsv2(d2)
        p2 = F.interpolate(p2, size=input_size, mode='bilinear', align_corners=True)
        
        p1 = self.dsv1(d1)
        p1 = F.interpolate(p1, size=input_size, mode='bilinear', align_corners=True)

        # Classification output
        class_outputs = self.classifier_head(x4)

        # Final output decision per sample
        batch_size = x.shape[0]
        classification_final = []
        segmentation_final = []
    
        for i in range(batch_size):
            if topk_idx[i, 0] == 0:  
                classification_final.append(class_outputs[i].unsqueeze(0))
                segmentation_final.append(None)
            else: 
                classification_final.append(None)
                segmentation_final.append([
                    p4[i].unsqueeze(0),
                    p3[i].unsqueeze(0),
                    p2[i].unsqueeze(0),
                    p1[i].unsqueeze(0),
                    final[i].unsqueeze(0)
                ])
    
        if self.training:
            return {
                'segmentation': segmentation_final,
                'classification': classification_final,
                'routing_weights': routing_weights,
                'active_expert': topk_idx,
                'deep_supervision': [p1, p2, p3, p4, final],
                'only_classification': class_outputs
            }
        else:
            return {
                'segmentation': segmentation_final,
                'classification': classification_final,
                'routing_weights': routing_weights,
                'active_expert': topk_idx,
                'only_classification': class_outputs
            }

#net = ResUNet(in_channels=3, num_classes=5, pretrained=True)
#from torchsummary import summary
#net.to('cuda' if torch.cuda.is_available() else 'cpu') 
#summary(net, input_size=(3, 224, 224))
