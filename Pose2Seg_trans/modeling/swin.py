from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "Kevin Wang"


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torchvision
import matplotlib.pyplot as plt
#try 1 import model directly from timm
import timm # dont forget


def visualise_feature_output(t):
  a = t[0].transpose(0,2)
  b = a.sum(-1)
  plt.imshow(b.detach().numpy())
  plt.show()

def swinv2_large(pretrained=True):
    model = timm.create_model('swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', pretrained=pretrained,features_only=True) #
    return model

def swinv2_large384(pretrained=True):
    model = timm.create_model('swinv2_large_window12to24_192to384', pretrained=pretrained,features_only=True) #
    return model
    

# fpn(feature pyramid extracter) wrapper for swim
class fpn(nn.Module):
  def __init__(self, model: timm.models):
    super(fpn, self).__init__()
    self.model = model
    channel_num = self.model.feature_info.channels()
    # c1, c2, c3, c4 all conv to same channel, fpn layers
    self.fpn_c4p4 = nn.Conv2d(channel_num[3], 256, kernel_size=1, stride=1, padding=0) 
    self.fpn_c3p3 = nn.Conv2d(channel_num[2], 256, kernel_size=1, stride=1, padding=0)
    self.fpn_c2p2 = nn.Conv2d(channel_num[1], 256, kernel_size=1, stride=1, padding=0)
    self.fpn_c1p1 = nn.Conv2d(channel_num[0], 256, kernel_size=1, stride=1, padding=0)

    self.fpn_p4 = nn.Conv2d( 256, 256, kernel_size=3, stride=1, padding=1)
    self.fpn_p3 = nn.Conv2d( 256, 256, kernel_size=3, stride=1, padding=1)
    self.fpn_p2 = nn.Conv2d( 256, 256, kernel_size=3, stride=1, padding=1)
    self.fpn_p1 = nn.Conv2d( 256, 256, kernel_size=3, stride=1, padding=1)


  def _upsample_add(self, x, y):
    '''Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    '''
    _,_,H,W = y.size()
    return F.upsample(x, size=(H,W), mode='bilinear') + y # 'bilinear'

  def forward(self, x):
    c1,c2,c3,c4=self.model(x) #feature extracted from the model
    p4 = self.fpn_c4p4(c4.transpose(1,3).transpose(2,3))
    p3 = self._upsample_add(p4, self.fpn_c3p3(c3.transpose(1,3).transpose(2,3)))
    p2 = self._upsample_add(p3, self.fpn_c2p2(c2.transpose(1,3).transpose(2,3)))
    p1 = self._upsample_add(p2, self.fpn_c1p1(c1.transpose(1,3).transpose(2,3)))
    # p4 = self.fpn_p4(p4)
    # p3 = self.fpn_p3(p3)
    # p2 = self.fpn_p2(p2)
    p1 = self.fpn_p1(p1)
    #to couple with the later section
    # p1 = F.interpolate(p1, size= (128,128), mode='nearest')
    return [p1,p2,p3,p4]




