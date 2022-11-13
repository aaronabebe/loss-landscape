import os

import timm
import torch
from torch.nn import functional as F
from timm.models import register_model, ResNet, Bottleneck
from timm.models.vision_transformer import VisionTransformer

import cifar10.models.densenet as densenet
import cifar10.models.resnet as resnet
import cifar10.models.vgg as vgg


@register_model
def resnet50_cifar10(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError('No pretrained ResNets :-(')

    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=10, **kwargs)


@register_model
def vit_tiny_cifar10(pretrained=False, **kwargs):
    """
    ViT-Tiny (Vit-T/8) for CIFAR10 training
    if pretrained = True then returns a Vit-T/16 pretrained on ImageNet with size 224.
    """
    if pretrained:
        return timm.create_model('vit_tiny_patch16_224', img_size=32, num_classes=10, pretrained=pretrained, **kwargs)
    return VisionTransformer(img_size=32, patch_size=8, num_classes=10, embed_dim=192, depth=12, num_heads=3, **kwargs)


# map between model name and function
models = {
    'vgg9': vgg.VGG9,
    'densenet121': densenet.DenseNet121,
    'resnet18': resnet.ResNet18,
    'resnet18_noshort': resnet.ResNet18_noshort,
    'resnet34': resnet.ResNet34,
    'resnet34_noshort': resnet.ResNet34_noshort,
    'resnet50': resnet.ResNet50,
    'resnet50_noshort': resnet.ResNet50_noshort,
    'resnet101': resnet.ResNet101,
    'resnet101_noshort': resnet.ResNet101_noshort,
    'resnet152': resnet.ResNet152,
    'resnet152_noshort': resnet.ResNet152_noshort,
    'resnet20': resnet.ResNet20,
    'resnet20_noshort': resnet.ResNet20_noshort,
    'resnet32_noshort': resnet.ResNet32_noshort,
    'resnet44_noshort': resnet.ResNet44_noshort,
    'resnet50_16_noshort': resnet.ResNet50_16_noshort,
    'resnet56': resnet.ResNet56,
    'resnet56_noshort': resnet.ResNet56_noshort,
    'resnet110': resnet.ResNet110,
    'resnet110_noshort': resnet.ResNet110_noshort,
    'wrn56_2': resnet.WRN56_2,
    'wrn56_2_noshort': resnet.WRN56_2_noshort,
    'wrn56_4': resnet.WRN56_4,
    'wrn56_4_noshort': resnet.WRN56_4_noshort,
    'wrn56_8': resnet.WRN56_8,
    'wrn56_8_noshort': resnet.WRN56_8_noshort,
    'wrn110_2_noshort': resnet.WRN110_2_noshort,
    'wrn110_4_noshort': resnet.WRN110_4_noshort,
}


def load(model_name, model_file=None, data_parallel=False):
    if 'custom' in model_name:
        if 'res' in model_name:
            net = timm.create_model('resnet50_cifar10')
        elif 'vit' in model_name:
            net = timm.create_model('vit_tiny_cifar10')
    else:
        net = models[model_name]()
    if data_parallel:  # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in stored.keys():
            # remove prefix from pl model
            net.load_state_dict(
                {k[len('model.'):]: v for k, v in stored['state_dict'].items() if k.startswith('model.')})
        else:
            net.load_state_dict(stored)

    if data_parallel:  # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
