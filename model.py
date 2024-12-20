import resnet
import torch
import torch.nn as nn
from typing import Dict, Iterable, Callable
import math

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True, from_feat=True
    ):
        super().__init__()
        self.from_feat=from_feat
        img_size = (img_size,img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if self.from_feat:
            return x
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        


    
    
    
    def forward(self, x):
        # Attention
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feedforward
        mlp_out = self.mlp(x)
        x = x + self.dropout(mlp_out)
        x = self.norm2(x)

        return x

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

 
    
class Transformer_Classifier(nn.Module):
    def __init__(self, image_size, input_dim, patch_size=14, num_classes=200, embed_dim=64, 
                 num_heads=4, mlp_dim=128, num_layers=6, dropout=0.1, from_feat=True):
        super(Transformer_Classifier, self).__init__()
        self.embed_dim=embed_dim
        self.dim_adapt_conv=nn.Conv2d(in_channels=input_dim, out_channels=self.embed_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.num_patches=(image_size//patch_size)**2
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches+1, embed_dim))
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.segment_vector = nn.Parameter(torch.ones(embed_dim))
        

    
    def forward(self, x):
        
        """
        Args:
            x: Tensor of shape (batch_size, H, W, input_dim), 
               where num_patches is the sequence length, and input_dim is the feature dimension.
        """
        N, C_input, H, W = x.size()
        x_input=x.clone()
        # Project input features to the embedding dimension
        x=self.dim_adapt_conv(x)
        C=self.embed_dim
        x_num_patch=H*W
        
        x=(torch.reshape(x, (N, C, x_num_patch)))
        x=x.permute(0,2,1) # reshape data to (batch_size, num_patches, embed_dim)

        # Add class token
        
        cls_token = self.cls_token.expand(N, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        #check if the input num of patch equal to the model num of patches
        if self.num_patches != x_num_patch:
            patch_pos_emb=self.pos_embed[:,1:,:]
            cls_pos_emb=self.pos_embed[:,0:1,:]
            dim_orig=int(math.sqrt(self.num_patches))
            dim_x=int(math.sqrt(x_num_patch))
            assert x_num_patch==dim_x*dim_x
            #interpolate the embedding pose
            x_pos_embed = torch.nn.functional.interpolate(
                patch_pos_emb.reshape(1, dim_orig, dim_orig, self.embed_dim).permute(0, 3, 1, 2), size=(dim_x, dim_x), mode='bicubic'
            )
            assert (dim_x, dim_x) == x_pos_embed.shape[-2:]
            x_pos_embed = x_pos_embed.permute(0, 2, 3, 1).view(1, -1, self.embed_dim)
            x_pos_embed=torch.cat((cls_pos_emb, x_pos_embed), dim=1)
                    
        else:
            x_pos_embed= self.pos_embed     

        # Add positional embedding
        x = x + x_pos_embed
        x = self.dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Classification head
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Take the class token
        seg_tokens=x[:,1:]
        cls_logits = self.head(cls_token_final)
        
        seg_prob=torch.sigmoid(torch.sum(seg_tokens*self.segment_vector, dim=-1))

        ccam_ = seg_prob.reshape(N, 1, H * W)                      # [N, 1, H*W]
        seg_prob=seg_prob.reshape(N,1, H, W)
        x_out = x_input.reshape(N, C_input, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x_out) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x_out) / (H * W)            # [N, 1, C]

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), seg_prob, cls_logits

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
    def __init__(self, image_size, pretrained='mocov2', cin=None, backbone_name='resnet'):
        super(Network, self).__init__()
        self.backbone_name=backbone_name
        if backbone_name=='resnet':
            self.backbone = ResNetSeries(pretrained=pretrained)
        else:
            dinov2_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
            self.backbone = FeatureExtractor(dinov2_model, layers=["norm"])


        #defining transformer classifier
        self.ac_head = Transformer_Classifier(image_size, cin)
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
        fg_feats, bg_feats, ccam, class_logits = self.ac_head(feats)

        return fg_feats, bg_feats, ccam, class_logits

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


def get_model(img_size, pretrained, cin=None):
    return Network(img_size, pretrained=pretrained, cin=cin, backbone_name='dinov2')
