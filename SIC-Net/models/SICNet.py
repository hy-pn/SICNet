from pyexpat.errors import XML_ERROR_SYNTAX
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from torchsummary import summary

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet50(pretrained,replace_stride_with_dilation=[False,True,True])     
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.conv = nn.Sequential(nn.Conv2d(2048+1024, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        initialize_weights(self.conv)
                                  
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + out
        out = self.relu(out)

        return out
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class Spatial_Attention(nn.Module):

    def __init__(self, kernel_size):
        super(Spatial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    
    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return mask



class DGM(nn.Module):
    def __init__(self,inplanes,planes):
        super(DGM,self).__init__()
                
        self.conv = nn.Conv2d(planes,inplanes,(1,1),1,bias=True)#     
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(inplanes+planes,inplanes,(1,1),stride=1,bias=True)
        self.sigmoid1 = nn.Sigmoid()

        self.conv1x1_2 = nn.Conv2d(inplanes,1,(1,1),stride=1,bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid2 = nn.Sigmoid()      
        self.sigmoid3 = nn.Sigmoid() 
        self.conv1x1_3 = nn.Conv2d(planes, inplanes,(1,1),1,bias=True)
              
    def forward(self,x,y):
        h1,w1 = x.size()[2:]
        h2,w2 = y.size()[2:]

        if h1==h2 and w1==w2:
            y0 = y
            y1 = self.conv1x1_3(y)
        else:
            y0 = y
            y1 = F.interpolate(y, size=(h1, w1), mode="bilinear", align_corners=True)    
            y1 = self.conv(y1)     
        x1 = x       
        x2 = self.conv1x1_2(x1)
        x2_h = self.pool1(x2)
        x2_w = self.pool2(x2)
        x2_h = self.sigmoid2(x2_h)
        x2_w = self.sigmoid3(x2_w)      
        x3 = self.avgpool1(x1)
        y2 = self.avgpool2(y0)
        
        y3 = torch.cat((x3,y2),1)
        y3 = self.conv1x1_1(y3)
        y3 = self.sigmoid1(y3)

        y1_1 = y3*y1
        y1_ = x2_h*y1_1+x2_w*y1_1


        y1_out = y1_+x1
        return F.relu(y1_out) 
class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        s = 4
        self.se1 =  DGM(64,128*s)
        self.se2 =DGM(64,256*s)
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        initialize_weights(self.se1,self.se1,self.conv1,self.conv2)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        # feat = self.conv3(feat)
        # feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params



class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = self._make_layers_(in_channels, out_channels)
        self.cb = ConvBlock(out_channels)

    def _make_layers_(self, in_channels, out_channels, blocks=2, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels))
        layers = [ResBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = self.cb(x)
        return x

class SICNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super(SICNet, self).__init__()
        s = 4
        self.FCN = FCN(in_channels, pretrained=True)
        self.classifier1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                             nn.Conv2d(64, num_classes, kernel_size=1))
        self.classifier2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                             nn.Conv2d(64, num_classes, kernel_size=1))
        self.sp = SpatialPath()
        self.downsampele2 = nn.Conv2d(64,128*s,1,2,0)
        self.downsampele3 = nn.Conv2d(64,256*s,1,2,0)

        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        self.SSCONV1 = DecoderBlock(512,128)
        self.SSCONV2 = DecoderBlock(128,128)
        self.CDCONV1 = DecoderBlock(1024+512,128)
        self.CDCONV2 = DecoderBlock(128,128)
        self.CDCONV3 = DecoderBlock(128,128)
        self.sigmoid = nn.Sigmoid()
        self.resCD = self._make_layer(ResBlock, 1024, 128, 4, stride=1)
        self.cd_classifier0 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                           nn.Conv2d(64, 1, kernel_size=1))
        
        self.cd_classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                           nn.Conv2d(64, 1, kernel_size=1))
        self.sa = Spatial_Attention(3)
        initialize_weights(self.resCD,self.cd_classifier0,self.SSCONV1,self.SSCONV2,self.CDCONV1,self.CDCONV2,self.CDCONV3,self.classifier1, self.classifier2, self.downsampele2,self.downsampele3)
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        x_s1 = self.sp.conv1(x)
        x_s2 = self.sp.conv2(x_s1)
        x0 = self.FCN.layer0(x) #size:1/4
        x0 = self.FCN.maxpool(x0) #size:1/4
        x1 = self.FCN.layer1(x0) #size:1/4
        x2 = self.FCN.layer2(x1) #size:1/8
        x_s3 = self.sp.se1(x_s2,x2)
        x2 = x2 + self.downsampele2(x_s3)
        x3 = self.FCN.layer3(x2) #size:1/16
        x_s4 = self.sp.se2(x_s3,x3)
        x3 = x3 + self.downsampele3(x_s4)
        x4 = self.FCN.layer4(x3)
        x4 = torch.cat([x3, x4], 1)
        x4 = self.FCN.conv(x4)
        return x4
    
    
    def forward(self, x1, x2):
        x_size = x1.size()
        x1 = self.base_forward(x1)
        x2 = self.base_forward(x2)
        x = torch.cat([x1,x2], 1)
        cd1 = self.resCD(x)
        change_attenion = self.sa(cd1)
        change1 = self.cd_classifier0(cd1)

        x1_h1 = self.SSCONV1(x1)
        x2_h1 = self.SSCONV1(x2)
        x1_h2 = x1_h1.permute(0,2,3,1)
        x2_h2 = x2_h1.permute(0,2,3,1)
        x1_out = change_attenion+x1_h1
        x2_out = change_attenion+x2_h1
        x1_out = self.SSCONV2(x1_out)
        x2_out = self.SSCONV2(x2_out)

        dist = F.pairwise_distance(x1_h2,x2_h2,keepdim=True)
        dist = self.sigmoid(dist)
        dist = dist.permute(0,3,1,2)

        cd = torch.cat([x1, x2, torch.abs(x1 - x2)], 1)
        x_a = self.CDCONV1(cd)
        x_a = x_a*dist+x_a
        x_a = self.upsample(x_a)
        out = self.CDCONV2(self.CDCONV3(x_a)+self.upsample(cd1))
        change = self.cd_classifier(out)

        out1 = self.classifier1(x1_out)
        out2 = self.classifier2(x2_out)
        
        return F.upsample(change1, x_size[2:], mode='bilinear'),F.upsample(change, x_size[2:], mode='bilinear'),F.upsample(out1, x_size[2:], mode='bilinear'), F.upsample(out2, x_size[2:], mode='bilinear')
