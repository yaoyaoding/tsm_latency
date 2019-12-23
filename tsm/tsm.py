import torch
import os
import torch.nn
import json
import numpy as np
import logging
from models import load_module
from thop import profile
from tvm_stack.utils import torch_latency_on_mpu, torch_latency_on_cpu, torch_latency_on_gpu, get_tracker, \
    torch_latency_on_nano, torch_error_on_nano
import pandas as pd

logging.disable(logging.WARNING)


def keep_middle(arr, keep_num=None):
    if keep_num is None:
        keep_num = (len(arr) + 1) // 2
    mean = np.mean(arr)
    arr = sorted(arr, key=lambda k: np.abs(k - mean))
    return arr[:keep_num]


def torch_parameters(torch_module):
    assert isinstance(torch_module, torch.nn.Module)
    return sum(p.numel() for p in torch_module.parameters() if p.requires_grad)


def test_error():
    rebuild = True
    number = 3
    repeat = 1
    nets = [
        ('TSM_MobileNetV2_1.0', load_module('TSM_MobileNetV2_1.0')),
    ]
    devices = [
        #        'mpu_pixel',
        #        'mpu_pixel2',
        #        'cpu',
        #        'gpu'
        'nano',
    ]
    opt_levels = [3]
    df = pd.DataFrame(columns=['Model', 'Resolution', 'Device', 'Opt-Level', 'Error'])
    index = 0
    results = []
    verbose = True
    failed = []
    for model_name, net in nets:
        for dummy_inputs in [
            (torch.rand(1, 3, 224, 224), torch.zeros(1, 12, 14, 14), torch.zeros(1, 20, 7, 7), torch.zeros(1, 20, 7, 7)),
            (torch.rand(1, 3, 112, 112), torch.zeros(1, 12, 7, 7), torch.zeros(1, 20, 4, 4), torch.zeros(1, 20, 4, 4))
        ]:
            for device in devices:
                for opt_level in opt_levels:
                    try:
                        if device.startswith('mpu'):
                            if device.endswith('pixel'):
                                device_key = 'p1'
                            else:
                                device_key = 'p2'
                            tracker = get_tracker()

                            pixel_config = {
                                'target': 'llvm -target=arm64-linux-android',
                                'target_host': 'llvm -target=arm64-linux-android'
                            }
                            latency_result = torch_latency_on_mpu(model_name, net, dummy_inputs, tracker,
                                                                  device_key=device_key, **pixel_config, number=number,
                                                                  opt_level=opt_level,
                                                                  repeat=repeat,
                                                                  rebuild=rebuild,
                                                                  verbose=verbose)
                        elif device == 'cpu':
                            latency_result = torch_latency_on_cpu(model_name, net, dummy_inputs,
                                                                  number=number,
                                                                  repeat=repeat,
                                                                  opt_level=opt_level,
                                                                  rebuild=rebuild,
                                                                  verbose=verbose)
                        elif device == 'gpu':
                            latency_result = torch_latency_on_gpu(model_name, net, dummy_inputs,
                                                                  number=number,
                                                                  repeat=repeat,
                                                                  opt_level=opt_level,
                                                                  rebuild=rebuild,
                                                                  verbose=verbose)
                        elif device == 'nano':
                            tracker = get_tracker()
                            error_result = torch_error_on_nano(model_name, net, dummy_inputs, tracker,
                                                               number=number,
                                                               repeat=repeat,
                                                               rebuild=rebuild,
                                                               verbose=verbose)
                    except Exception as e:
                        raise e
                        failed.append((model_name, device, str(e)))
                        print(e)
                        continue
                    print(error_result)
                    df.loc[len(df)] = [model_name, dummy_inputs[0].size()[2], device, opt_level, np.mean(error_result)]
                    print(df.to_string(index=False))
                    index += 1
    print(df.to_string(index=False))
    if len(failed) > 0:
        print(failed)
    os.chdir(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
    with open('summary.txt', 'wt') as f:
        f.write(df.to_string(index=False))


def test_all():
    rebuild = True
    number = 3
    repeat = 100
    nets = [
        ('TSM_MobileNetV2_1.0', load_module('TSM_MobileNetV2_1.0')),
    ]
    devices = [
        #        'mpu_pixel',
        #        'mpu_pixel2',
        #        'cpu',
        #        'gpu'
        'nano',
    ]
    opt_levels = [3]
    df = pd.DataFrame(columns=['Model', 'Resolution', 'Device', 'Opt-Level', 'FrameRate(Hz)', 'Latency(ms)', 'Std(ms)'])
    index = 0
    results = []
    verbose = True
    failed = []
    for model_name, net in nets:
        for dummy_inputs in [
            (
            torch.rand(1, 3, 224, 224), torch.zeros(1, 12, 14, 14), torch.zeros(1, 20, 7, 7), torch.zeros(1, 20, 7, 7)),
            (torch.rand(1, 3, 112, 112), torch.zeros(1, 12, 7, 7), torch.zeros(1, 20, 4, 4), torch.zeros(1, 20, 4, 4))
        ]:
            for device in devices:
                for opt_level in opt_levels:
                    try:
                        if device.startswith('mpu'):
                            if device.endswith('pixel'):
                                device_key = 'p1'
                            else:
                                device_key = 'p2'
                            tracker = get_tracker()

                            pixel_config = {
                                'target': 'llvm -target=arm64-linux-android',
                                'target_host': 'llvm -target=arm64-linux-android'
                            }
                            latency_result = torch_latency_on_mpu(model_name, net, dummy_inputs, tracker,
                                                                  device_key=device_key, **pixel_config, number=number,
                                                                  opt_level=opt_level,
                                                                  repeat=repeat,
                                                                  rebuild=rebuild,
                                                                  verbose=verbose)
                        elif device == 'cpu':
                            latency_result = torch_latency_on_cpu(model_name, net, dummy_inputs,
                                                                  number=number,
                                                                  repeat=repeat,
                                                                  opt_level=opt_level,
                                                                  rebuild=rebuild,
                                                                  verbose=verbose)
                        elif device == 'gpu':
                            latency_result = torch_latency_on_gpu(model_name, net, dummy_inputs,
                                                                  number=number,
                                                                  repeat=repeat,
                                                                  opt_level=opt_level,
                                                                  rebuild=rebuild,
                                                                  verbose=verbose)
                        elif device == 'nano':
                            tracker = get_tracker()
                            latency_result = torch_latency_on_nano(model_name, net, dummy_inputs, tracker,
                                                                   number=number,
                                                                   repeat=repeat,
                                                                   rebuild=rebuild,
                                                                   verbose=verbose)
                    except Exception as e:
                        failed.append((model_name, device, str(e)))
                        print(e)
                        continue

                    latency_result = np.array(keep_middle(latency_result)) * 1000.0
                    mean_latency = np.mean(latency_result)
                    std_latency = np.std(latency_result)
                    df.loc[len(df)] = [model_name, dummy_inputs[0].size()[2], device, opt_level, 1000 / mean_latency,
                                       mean_latency, std_latency]
                    results.append({
                        'model_name': model_name,
                        'device': device,
                        'opt_level': opt_level,
                        'mean_latency(ms)': mean_latency,
                        'std_latency(ms)': std_latency,
                        'latency': latency_result.tolist()
                    })
                    print(df.to_string(index=False))
                    index += 1
    print(df.to_string(index=False))
    if len(failed) > 0:
        print(failed)
    os.chdir(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
    json.dump(results, open('results.all.json', 'wt'), indent=2)
    json.dump(failed, open('failed.json', 'wt'), indent=2)
    with open('summary.txt', 'wt') as f:
        f.write(df.to_string(index=False))


test_all()
