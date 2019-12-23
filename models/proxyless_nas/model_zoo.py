from functools import partial
import copy
import os
import json

import torch

from models.proxyless_nas.utils import download_url
from .nas_modules import ProxylessNASNets


def proxyless_base(pretrained=True, net_config=None, net_weight=None):
    assert net_config is not None, "Please input a network config"
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))
    net = ProxylessNASNets.build_from_config(net_config_json)

    if 'bn' in net_config_json:
        net.set_bn_param(
            bn_momentum=net_config_json['bn']['momentum'],
            bn_eps=net_config_json['bn']['eps'])
    else:
        net.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init_path = download_url(net_weight)
        init = torch.load(init_path, map_location='cpu')
        net.load_state_dict(init['state_dict'])

    return net


def proxyless_from_config(net_config_path):
    net_config_json = json.load(open(net_config_path, 'r'))
    net = ProxylessNASNets.build_from_config(net_config_json)
    if 'bn' in net_config_json:
        net.set_bn_param(
            bn_momentum=net_config_json['bn']['momentum'],
            bn_eps=net_config_json['bn']['eps'])
    return net



proxyless_cpu = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_cpu.pth")

proxyless_gpu = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_gpu.pth")

proxyless_mobile = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile.pth")

proxyless_mobile_14 = partial(
    proxyless_base,
    net_config="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.config",
    net_weight="https://hanlab.mit.edu/files/proxylessNAS/proxyless_mobile_14.pth")


def proxyless_test():
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "proxyless_gpu.config")
    net_config_json = json.load(open(config_path, 'r'))
    net = ProxylessNASNets.build_from_config(net_config_json)

    if 'bn' in net_config_json:
        net.set_bn_param(
            bn_momentum=net_config_json['bn']['momentum'],
            bn_eps=net_config_json['bn']['eps'])
    else:
        net.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)

    return net

def proxyless_tests():
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "proxyless_test.config")
    net_config_json = json.load(open(config_path, 'r'))
    raw_config = copy.deepcopy(net_config_json)
    torch_modules = []
    blen = len(raw_config['blocks'])
    for l in range(1, blen+1):
        for start in range(blen-l+1):
            end = start + l - 1
            config = copy.deepcopy(raw_config)
            config['blocks'] = config['blocks'][start:end+1]
            first_block = config['blocks'][0]
            if 'in_channels' in first_block['mobile_inverted_conv']:
                in_channels = first_block['mobile_inverted_conv']['in_channels']
            elif 'in_channels' in first_block['shortcut']:
                in_channels = first_block['shortcut']['in_channels']
            else:
                raise ValueError
            last_block = config['blocks'][-1]
            if 'out_channels' in last_block['mobile_inverted_conv']:
                out_channels = last_block['mobile_inverted_conv']['out_channels']
            elif 'out_channels' in last_block['shortcut']:
                out_channels = last_block['shortcut']['out_channels']
            else:
                raise ValueError
            config['first_conv']['out_channels'] = in_channels
            config['feature_mix_layer']['in_channels'] = out_channels
            torch_modules.append((f'proxyless_{start}_{end}_test', ProxylessNASNets.build_from_config(config)))
    return torch_modules



