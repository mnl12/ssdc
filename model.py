import resnet
import torch
import torch.nn as nn
from typing import Dict, Iterable, Callable


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetSeries(nn.Module):
    def __init__(self, pretrained):
        super(ResNetSeries, self).__init__()

        if pretrained == 'supervised':
            print(f'Loading supervised pretrained parameters!')
            model = resnet.resnet50(pretrained=True)
        elif pretrained == 'mocov2':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet.resnet50(pretrained=False)
            checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif pretrained == 'detco':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet.resnet50(pretrained=False)
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)

        return torch.cat([x2, x1], dim=1)
        #return x2

class Distillator(nn.Module):
    def __init__(self, cin):
        super(Distillator, self).__init__()
        self.conv2d = nn.Conv2d(cin, 500, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.relu_activation=nn.ReLU()
    
    def forward(self, x):
        output=self.relu_activation(self.conv2d(x))
        return output
    
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x    
    
class Disentangler(nn.Module):
    def __init__(self, cin):
        super(Disentangler, self).__init__()
        mid_layer_dim=512
        mid_layer_dim2=256
        self.relu_func=nn.ReLU()

        self.activation_head1 = nn.Conv2d(cin, mid_layer_dim, kernel_size=3, padding=1, bias=True)
        self.activation_head2 = nn.Conv2d(mid_layer_dim, mid_layer_dim2, kernel_size=3, padding=1, bias=True)
        self.activation_head3 = nn.Conv2d(mid_layer_dim2, 1, kernel_size=1, padding=0, bias=True)


        #torch.nn.init.xavier_uniform(self.activation_head.weight)

        self.bn_head3 = nn.BatchNorm2d(1)
        self.bn_head1 = nn.BatchNorm2d(mid_layer_dim)
        self.bn_head2 = nn.BatchNorm2d(mid_layer_dim2)

    def forward(self, x):
        N, C, H, W = x.size()
        x1=self.relu_func(self.bn_head1(self.activation_head1(x)))
        x2=self.relu_func(self.bn_head2(self.activation_head2(x1)))
        ccam = torch.sigmoid(self.bn_head3(self.activation_head3(x2)))
        #ccam = torch.sigmoid(self.bn_head(self.activation_head2(x)))


        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features


class Network(nn.Module):
    def __init__(self, pretrained='mocov2', cin=None, backbone_name='resnet'):
        super(Network, self).__init__()
        self.backbone_name=backbone_name
        if backbone_name=='resnet':
            self.backbone = ResNetSeries(pretrained=pretrained)
        else:
            dinov2_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
            self.backbone = FeatureExtractor(dinov2_model, layers=["norm"])


        #self.distiller=Distillator(cin)
        #self.print_layer=PrintLayer()
        self.ac_head = Disentangler(cin)
        self.from_scratch_layers = [self.ac_head]

    def forward(self, x):
        
        if self.backbone_name=="resnet":
            feats = self.backbone(x)
            #print("features shapes are", feats.shape)

        else:
            N1, C1, H1, W1 = x.size()
            features=self.backbone(x)
            seg_features=features['norm'][:,1:,:].detach()
            feats=(torch.reshape(seg_features, (seg_features.shape[0], int(H1/14), int(W1/14), seg_features.shape[2])))
            feats=feats.permute(0,3,1,2)
            #print("features shapes are", feats.shape)
        fg_feats, bg_feats, ccam = self.ac_head(feats)

        return fg_feats, bg_feats, ccam

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups


def get_model(pretrained, cin=None):
    return Network(pretrained=pretrained, cin=cin, backbone_name='dinov2')
