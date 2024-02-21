import torch
import torchvision
from torch import nn
import torch.nn.functional as func

class BasicBlock(nn.Module):
  expansion = 1 #the number of output channels is x times the number of input channels
  def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
    super(BasicBlock, self).__init__() #initialization of the parent class
    self.conv1=nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                         kernel_size=3, stride=stride, padding=1, bias=False) # in_channels and out_channels are required, stride has default value of 1
    self.bn1 = nn.BatchNorm2d(out_channel) #specify only the number of output channels, and the layer takes care of the normalization for each channel in the input data during training
    self.relu = torch.nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                           kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channel)
    self.downsample = downsample
    #torch.nn.Linear(100, 200), torch.nn.Softmax()

  def forward(self, x):
    identity = x
    if self.downsample is not None: #1*1 convolution and batch normalization
        identity = self.downsample(x)
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += identity
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self,block,blocks_num,num_classes=1000,include_top=True,groups=1,width_per_group=64,input_img_size=(704, 256),pre_trained=False):
    #Conditional Top Layers: If include_top is set to True, an adaptive average pooling layer (avgpool) and a fully connected layer (fc) are added at the end of the network
    #num_classes: Number of output classes (default is set to 1000 for ImageNet)
    #groups and width_per_group: used when creating the blocks in each layer, controlling the number of groups and width of each group
    super(ResNet, self).__init__()
    self.include_top = include_top
    self.in_channel = 64
    self.groups = groups
    self.width_per_group = width_per_group
    self.input_img_size = input_img_size
    self.num_classes = num_classes
    self.pre_trained = pre_trained

    if pre_trained:
      res=torchvision.models.resnet18(weights='DEFAULT') 
      self.__dict__.update(res.__dict__)
      self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
    else:
      self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                                padding=3, bias=False)
      self.bn1 = nn.BatchNorm2d(self.in_channel)
      self.relu = nn.ReLU(inplace=True)
      self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      self.layer1 = self._make_layer(block, 64, blocks_num[0])
      self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
      self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
      self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
      if self.include_top: #whether to inlcude fc in the last layer for classification, true=ap+linear, otherwise the network only get features
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
          self.fc = nn.Linear(512 * block.expansion, num_classes)
      for m in self.modules(): #apply kaiming approach to cov initialization
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def _make_layer(self, block, channel, block_num, stride=1):
      downsample = None
      if stride != 1 or self.in_channel != channel * block.expansion:
          downsample = nn.Sequential(
              nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(channel * block.expansion))

      layers = []
      layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample, #the first layer might need to be downsampled
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
      self.in_channel = channel * block.expansion

      for _ in range(1, block_num): #range(1,2)=1
          layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

      return nn.Sequential(*layers) #as layer 1,2..


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    feature_maps = x
    if self.include_top:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        scores = self.fc(x)
    if self.training:
        return scores
    else: 
       probs = nn.functional.softmax(scores, dim=-1)
       weights = self.fc.weight
       weights = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat((
                x.size(0),1,1,self.input_img_size[0] // 2**5,self.input_img_size[1] // 2**5,)))
       feature_maps = feature_maps.unsqueeze(1).repeat((1, probs.size(1), 1, 1, 1))
       location = torch.mul(weights, feature_maps).sum(axis=2)
       location = func.interpolate(location, size=self.input_img_size, mode="bilinear")
       maxs, _ = location.max(dim=-1, keepdim=True)
       maxs, _ = maxs.max(dim=-2, keepdim=True)
       mins, _ = location.min(dim=-1, keepdim=True)
       mins, _ = mins.min(dim=-2, keepdim=True)
       norm_location = (location - mins) / (maxs - mins)
       return probs, norm_location

  def resnet18(num_classes=1000, include_top=True, input_img_size=(704, 256),pre_trained=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, input_img_size=input_img_size, pre_trained=pre_trained)