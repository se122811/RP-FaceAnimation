import sys
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import numpy as np
import numpy as np
from einops import rearrange, repeat



class LayerNorm2d(nn.Module):
    def __init__(self, n_out, affine=True):
        super(LayerNorm2d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
          self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
          self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
          return F.layer_norm(x, normalized_shape, \
              self.weight.expand(normalized_shape), 
              self.bias.expand(normalized_shape))
              
        else:
          return F.layer_norm(x, normalized_shape)  


class ADAINHourglass(nn.Module):
    def __init__(self, image_nc, pose_nc, ngf, img_f, encoder_layers, decoder_layers, nonlinearity, use_spect):
        super(ADAINHourglass, self).__init__()
        self.encoder = ADAINEncoder(image_nc, pose_nc, ngf, img_f, encoder_layers, nonlinearity, use_spect)
        self.decoder = ADAINDecoder(pose_nc, ngf, img_f, encoder_layers, decoder_layers, True, nonlinearity, use_spect)
        self.output_nc = self.decoder.output_nc

    def forward(self, x, z):
        return self.decoder(self.encoder(x, z), z)                 



class ADAINEncoder(nn.Module):
    def __init__(self, image_nc, pose_nc, ngf, img_f, layers, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINEncoder, self).__init__()
        self.layers = layers
        self.input_layer = nn.Conv2d(image_nc, ngf, kernel_size=7, stride=1, padding=3)
        for i in range(layers):
            in_channels = min(ngf * (2**i), img_f)
            out_channels = min(ngf *(2**(i+1)), img_f)
            model = ADAINEncoderBlock(in_channels, out_channels, pose_nc, nonlinearity, use_spect)
            setattr(self, 'encoder' + str(i), model)
        self.output_nc = out_channels
        
    def forward(self, x, z):
        out = self.input_layer(x)
        out_list = [out]
        for i in range(self.layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out, z)
            out_list.append(out)
        return out_list
        
class ADAINDecoder(nn.Module):
    """docstring for ADAINDecoder"""
    def __init__(self, pose_nc, ngf, img_f, encoder_layers, decoder_layers, skip_connect=True, 
                 nonlinearity=nn.LeakyReLU(), use_spect=False):

        super(ADAINDecoder, self).__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.skip_connect = skip_connect
        use_transpose = True

        for i in range(encoder_layers-decoder_layers, encoder_layers)[::-1]:
            in_channels = min(ngf * (2**(i+1)), img_f)
            in_channels = in_channels*2 if i != (encoder_layers-1) and self.skip_connect else in_channels
            out_channels = min(ngf * (2**i), img_f)
            model = ADAINDecoderBlock(in_channels, out_channels, out_channels, pose_nc, use_transpose, nonlinearity, use_spect)
            setattr(self, 'decoder' + str(i), model)

        self.output_nc = out_channels*2 if self.skip_connect else out_channels

    def forward(self, x, z):
        out = x.pop() if self.skip_connect else x
        for i in range(self.encoder_layers-self.decoder_layers, self.encoder_layers)[::-1]:
            model = getattr(self, 'decoder' + str(i))
            out = model(out, z)
            out = torch.cat([out, x.pop()], 1) if self.skip_connect else out
        return out

class ADAINEncoderBlock(nn.Module):       
    def __init__(self, input_nc, output_nc, feature_nc, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINEncoderBlock, self).__init__()
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        self.conv_0 = spectral_norm(nn.Conv2d(input_nc,  output_nc, **kwargs_down), use_spect)
        self.conv_1 = spectral_norm(nn.Conv2d(output_nc, output_nc, **kwargs_fine), use_spect)


        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(output_nc, feature_nc)
        self.actvn = nonlinearity

    def forward(self, x, z):
        x = self.conv_0(self.actvn(self.norm_0(x, z)))
        x = self.conv_1(self.actvn(self.norm_1(x, z)))
        return x

class ADAINDecoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc, feature_nc, use_transpose=True, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINDecoderBlock, self).__init__()        
        # Attributes
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        kwargs_fine = {'kernel_size':3, 'stride':1, 'padding':1}
        if use_transpose:
            kwargs_up = {'kernel_size':3, 'stride':2, 'padding':1, 'output_padding':1}
        else:
            kwargs_up = {'kernel_size':3, 'stride':1, 'padding':1}

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, **kwargs_fine), use_spect)
        if use_transpose:
            self.conv_1 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, **kwargs_up), use_spect)
            self.conv_s = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, **kwargs_up), use_spect)
        else:
            self.conv_1 = nn.Sequential(spectral_norm(nn.Conv2d(hidden_nc, output_nc, **kwargs_up), use_spect),
                                        nn.Upsample(scale_factor=2))
            self.conv_s = nn.Sequential(spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs_up), use_spect),
                                        nn.Upsample(scale_factor=2))
        # define normalization layers
        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(hidden_nc, feature_nc)
        self.norm_s = ADAIN(input_nc, feature_nc)
        
    def forward(self, x, z):
        x_s = self.shortcut(x, z)
        dx = self.conv_0(self.actvn(self.norm_0(x, z)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, z)))
        out = x_s + dx
        return out

    def shortcut(self, x, z):
        x_s = self.conv_s(self.actvn(self.norm_s(x, z)))
        return x_s              


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out


class FineEncoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, image_nc, ngf, img_f, layers, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineEncoder, self).__init__()
        self.layers = layers
        self.first = FirstBlock2d(image_nc, ngf, norm_layer, nonlinearity, use_spect)
        
        for i in range(layers):
            in_channels = min(ngf*(2**i), img_f)
            out_channels = min(ngf*(2**(i+1)), img_f)
            model = DownBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            setattr(self, 'down' + str(i), model)
        self.output_nc = out_channels

    def forward(self, x):
        x = self.first(x)
        out=[x]
        for i in range(self.layers):
            model = getattr(self, 'down'+str(i))
            x = model(x)
            out.append(x)
        return out
    


class FineDecoder(nn.Module):
    """docstring for FineDecoder"""
    def __init__(self, image_nc, feature_nc, ngf, img_f, layers, num_block, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineDecoder, self).__init__()
        self.layers = layers
        for i in range(layers)[::-1]:
            in_channels = min(ngf*(2**(i+1)), img_f)
            out_channels = min(ngf*(2**i), img_f)
            up = UpBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            res = FineADAINResBlocks(num_block, in_channels, feature_nc, norm_layer, nonlinearity, use_spect)
            jump = Jump(out_channels, norm_layer, nonlinearity, use_spect)

            setattr(self, 'up' + str(i), up)
            setattr(self, 'res' + str(i), res)            
            setattr(self, 'jump' + str(i), jump)

        self.final = FinalBlock2d(out_channels, image_nc, use_spect, 'tanh')

        self.output_nc = out_channels

    def forward(self, x, z):
        out = x.pop()
        for i in range(self.layers)[::-1]:
            res_model = getattr(self, 'res' + str(i))
            up_model = getattr(self, 'up' + str(i))
            jump_model = getattr(self, 'jump' + str(i))
            out = res_model(out, z)
            out = up_model(out)
            out = jump_model(x.pop()) + out
        out_image = self.final(out)
        return out_image



class FineUVEncoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, image_nc, ngf, img_f, layers, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineUVEncoder, self).__init__()
        self.layers = layers
        self.first = FirstBlock2d(image_nc, ngf, norm_layer, nonlinearity, use_spect)
        
        for i in range(layers):
            in_channels = min(ngf*(2**i), img_f)
            out_channels = min(ngf*(2**(i+1)), img_f)
            model = DownBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            setattr(self, 'down' + str(i), model)
        self.output_nc = out_channels

    def forward(self, x):
        x = self.first(x)
        out=[x]
        for i in range(self.layers):
            model = getattr(self, 'down'+str(i))
            x = model(x)
            out.append(x)
        return out
    



class FineUVDecoder(nn.Module):
    """docstring for FineDecoder"""
    def __init__(self, image_nc, ngf, img_f, layers, num_block, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineUVDecoder, self).__init__()
        self.layers = layers
        
        for i in range(layers)[::-1]:
            in_channels = min(ngf*(2**(i+1)), img_f)
            out_channels = min(ngf*(2**i), img_f)
            
            up = UpBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            # res = FineADAINResBlocks(num_block, in_channels, feature_nc, norm_layer, nonlinearity, use_spect)
            jump = Jump(out_channels, norm_layer, nonlinearity, use_spect)


            setattr(self, 'up' + str(i), up)
            # setattr(self, 'res' + str(i), res)            
            setattr(self, 'jump' + str(i), jump)

        self.final = FinalBlock2d(out_channels, image_nc, use_spect, 'tanh')

        self.output_nc = out_channels

    def forward(self, x):
        out = x.pop()
        for i in range(self.layers)[::-1]:
            # res_model = getattr(self, 'res' + str(i))
            up_model = getattr(self, 'up' + str(i))
            jump_model = getattr(self, 'jump' + str(i))
            out = up_model(out)
            out = jump_model(x.pop()) + out
        out_image = self.final(out)
        return out_image



class FirstBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FirstBlock2d, self).__init__()
        kwargs = {'kernel_size': 7, 'stride': 1, 'padding': 3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        self.input_nc = input_nc
        self.output_nc = output_nc
        
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out  
    
    

class DownBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(DownBlock2d, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        pool = nn.AvgPool2d(kernel_size=(2, 2))

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity, pool)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity, pool)

    def forward(self, x):
        out = self.model(x)
        return out 


class UpBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(UpBlock2d, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)

    def forward(self, x):
        out = self.model(F.interpolate(x, scale_factor=2))
        return out


class FineADAINResBlocks(nn.Module):
    def __init__(self, num_block, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINResBlocks, self).__init__()                                
        self.num_block = num_block
        for i in range(num_block):
            model = FineADAINResBlock2d(input_nc, feature_nc, norm_layer, nonlinearity, use_spect)
            setattr(self, 'res'+str(i), model)

    def forward(self, x, z):
        for i in range(self.num_block):
            model = getattr(self, 'res'+str(i))
            x = model(x, z)
        return x     

class Jump(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(Jump, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(input_nc), nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out          

class FineADAINResBlock2d(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINResBlock2d, self).__init__()

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        self.conv1 = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        self.conv2 = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        self.norm1 = ADAIN(input_nc, feature_nc)
        self.norm2 = ADAIN(input_nc, feature_nc)

        self.actvn = nonlinearity


    def forward(self, x, z):
        dx = self.actvn(self.norm1(self.conv1(x), z))
        dx = self.norm2(self.conv2(x), z)
        out = dx + x
        return out        

class FinalBlock2d(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, use_spect=False, tanh_or_sigmoid='tanh'):
        super(FinalBlock2d, self).__init__()

        kwargs = {'kernel_size': 7, 'stride': 1, 'padding':3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        if tanh_or_sigmoid == 'sigmoid':
            out_nonlinearity = nn.Sigmoid()
        else:
            out_nonlinearity = nn.Tanh()            

        self.model = nn.Sequential(conv, out_nonlinearity)
    def forward(self, x):
        out = self.model(x)
        return out    
    
    

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(256))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0),*256)
        return img



class SLN(nn.Module):
    """
    Self-modulated LayerNorm
    """
    def __init__(self, num_features):
        super(SLN, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        # self.gamma = nn.Parameter(torch.FloatTensor(1, 1, 1))
        # self.beta = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")
        self.beta = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")

    def forward(self, hl, w):
        return self.gamma * w * self.ln(hl) + self.beta * w


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".

    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """
    def __init__(self, dim, num_heads = 4, dim_head = None, discriminator = False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        self.weight_dim = self.num_heads * self.dim_head
        self.to_qkv = nn.Linear(dim, self.weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5
        self.discriminator = discriminator
        self.w_out = nn.Linear(self.weight_dim, dim, bias = True)

        if discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3

        if self.discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.to_qkv.weight = torch.nn.Parameter(self.to_qkv.weight * self.init_spect_norm / torch.max(s))

        # Generate the q, k, v vectors
        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k = 3, h = self.num_heads))

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.discriminator:
            attn = torch.cdist(q, k, p = 2)
        else:
            attn = torch.einsum("... i d, ... j d -> ... i j", q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim = -1)
        result = torch.einsum("... i j, ... j d -> ... i d", scale_attn_score, v)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.w_out(result)


class DEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(DEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head, discriminator = True)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attn(x1))
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class GEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(GEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = SLN(dim)
        self.norm2 = SLN(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, hl, x):
        hl_temp = self.dropout(self.attn(self.norm1(hl, x))) + hl
        hl_final = self.mlp(self.norm2(hl_temp, x)) + hl_temp
        return x, hl_final


class GTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(GTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(GEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, hl, x):
        for block in self.blocks:
            x, hl = block(hl, x)
        return x, hl


class DTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(DTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(DEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """
    def __init__(self, in_features, out_features, bias = True,is_first = False, omega_0 = 30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Transformer_decoder(nn.Module):
    def __init__(self,
        initialize_size = 32,
        dim = 384,
        blocks = 6,
        num_heads = 6,
        dim_head = None,
        dropout = 0,
        out_channels = 3
    ):
        super(Transformer_decoder, self).__init__()
        self.initialize_size = initialize_size
        self.dim = dim
        self.blocks = blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.out_channels = out_channels

        self.pos_emb1D = nn.Parameter(torch.randn(self.initialize_size * 8, dim))

        self.mlp = nn.Linear(1024, (self.initialize_size * 8) * self.dim)
        self.Transformer_Encoder = GTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)

        # Implicit Neural Representation
        self.w_out = nn.Sequential(
            SineLayer(dim, dim * 2, is_first = True, omega_0 = 30.),
            SineLayer(dim * 2, self.initialize_size * 8 * self.out_channels, is_first = False, omega_0 = 30)
        )
        self.sln_norm = SLN(self.dim)

    def forward(self, noise):
        
        # x = ([64,256,384])
        x = self.mlp(noise).view(-1, self.initialize_size * 8, self.dim)
        
        x, hl = self.Transformer_Encoder(self.pos_emb1D, x) # --> torch.Size([64, 256, 384])
        x = self.sln_norm(hl, x) #--> torch.Size([64, 256, 384])
        x = self.w_out(x)  # Replace to siren --> torch.Size([64, 256, 768])
        result = x.view(x.shape[0], 3, self.initialize_size * 8, self.initialize_size * 8)
        
        
        return result 



class Transformer_encoder(nn.Module):
    def __init__(self,
        in_channels = 6,
        patch_size = 8,
        extend_size = 2,
        dim = 384,
        blocks = 6,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Transformer_encoder, self).__init__()
        self.patch_size = patch_size + 2 * extend_size
        self.token_dim = in_channels * (self.patch_size ** 2)
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(self.token_dim + 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1024)
        )

        self.Transformer_Encoder = DTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)



    def forward(self, img, img2):
        # Generate overlappimg image patches
        
        img = torch.cat([img,img2], dim=1)
        
        stride_h = (img.shape[2] - self.patch_size) // 8 + 1
        stride_w = (img.shape[3] - self.patch_size) // 8 + 1
        
        img_patches = img.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        img_patches = img_patches.contiguous().view(
            img_patches.shape[0], img_patches.shape[2] * img_patches.shape[3], img_patches.shape[1] * img_patches.shape[4] * img_patches.shape[5]
        )
        img_patches = self.project_patches(img_patches)
        batch_size, tokens, _ = img_patches.shape

        # Prepend the classifier token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b = batch_size)
        img_patches = torch.cat((cls_token, img_patches), dim = 1)

        # Plus the positional embedding
        img_patches = img_patches + self.pos_emb1D[: tokens + 1, :]
        img_patches = self.emb_dropout(img_patches)

        result = self.Transformer_Encoder(img_patches)
        logits = self.mlp_head(result[:, 0, :])
        # logits = nn.Sigmoid()(logits)
        
        return logits


# def test_both():
#     B, dim = 10, 1024
#     G = Generator(initialize_size = 32, dropout = 0.1)
#     noise = torch.FloatTensor(np.random.normal(0, 1, (B, dim)))
#     fake_img = G(noise)
#     D = Discriminator(patch_size = 8, dropout = 0.1)
#     D_logits = D(fake_img)
#     print(D_logits)
#     print(f"Max: {torch.max(D_logits)}, Min: {torch.min(D_logits)}")
