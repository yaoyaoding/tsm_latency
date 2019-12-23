from __future__ import print_function
import os
import tvm.rpc as rpc
from functools import partial
import tvm
from tvm.contrib import cc
import pandas as pd
import logging
import tvm.relay
from tvm.contrib import graph_runtime
import onnx
import numpy as np
import io
import argparse
import torch
import torch.onnx
from models import load_module


# from layerwise_latency.utils import keep_middle_values
def keep_middle_values(arr, keep_num):
    mean = np.mean(arr)
    arr = sorted(arr, key=lambda k: np.abs(k - mean))
    return arr[:keep_num]


logging.disable(logging.WARNING)

number = 5
repeat = 100

argparser = argparse.ArgumentParser()
argparser.add_argument('-no_gpu', action='store_true', default=False)
argparser.add_argument('-device', type=str, required=True, choices=['pi', 'nano', 'tx2', 'pi3b'])

args = argparser.parse_args()


def torch2relay(module, dummy_inputs):
    input_shape = {}
    input_names = []
    module.eval()
    with torch.no_grad():
        for index, dummy_input in enumerate(dummy_inputs):
            if isinstance(dummy_input, np.ndarray):
                dummy_input = torch.from_numpy(dummy_input).detach()
            assert isinstance(dummy_input, torch.Tensor)
            input_name = 'input_{index}'.format(index=index)
            input_shape[input_name] = dummy_input.shape
            input_names.append(input_name)
        buffer = io.BytesIO()
        module.eval()
        torch.onnx.export(module, dummy_inputs, buffer, input_names=input_names)
        if not isinstance(buffer, str):
            buffer.seek(0, 0)
        onnx_model = onnx.load_model(buffer)
        return tvm.relay.frontend.from_onnx(onnx_model, shape=input_shape)


def test_latency(net, inputs, target, target_host, ctx):
    # ops, _ = thop.profile(net, inputs, verbose=False)

    print("torch -> relay")
    relay_module, params = torch2relay(net, inputs)

    print("relay -> tvm", "target =", target)
    with tvm.relay.build_config(opt_level=3):
        graph, tvm_module, params = tvm.relay.build(relay_module, target=target, target_host=target_host, params=params)

    print("create graph runtime")
    graph_module = graph_runtime.create(graph, tvm_module, ctx)

    print("run tvm module")
    ftimer = graph_module.module.time_evaluator('run', ctx, number, repeat)
    results = [float(v) * 1000.0 for v in ftimer().results]
    return results


def test_latency_rpc(net, inputs, target, target_host, device_key):
    print("torch -> relay")
    relay_module, params = torch2relay(net, inputs)


    print("relay -> tvm", "target =", target)
    with tvm.relay.build_config(opt_level=3):
        graph, tvm_module, params = tvm.relay.build(relay_module, target=target, target_host=target_host)

    lib_fname = '/tmp/mod.so'
    print(f"export lib {lib_fname}")
#    tvm_module.export_library(lib_fname, partial(cc.create_shared, cc="/home/yaoyao/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc"))
    tvm_module.export_library(lib_fname, partial(cc.create_shared, cc="/usr/bin/aarch64-linux-gnu-gcc"))

    print("request seesion from tracker")
    tracker = get_tracker()
    session = tracker.request(key=device_key, session_timeout=1000, max_retry=1)

    print("upload lib")
    ctx = session.cpu(0)
    session.upload(lib_fname)

    print("remote load lib")
    session_lib = session.load_module(os.path.basename(lib_fname))

    print("remote create runtime")
    tvm_module = graph_runtime.create(graph, session_lib, ctx)

    print("run")
    ftimer = tvm_module.module.time_evaluator('run', ctx, number=number, repeat=repeat)
    results = [float(v) * 1000.0 for v in ftimer().results]
    return results


def get_tracker(host=None, port=None):
    if host and port:
        return rpc.connect_tracker(host, port)
    else:
        host = os.environ.get('RPC_HOST')
        port = os.environ.get('RPC_PORT')
        return rpc.connect_tracker(host, int(port))


def main():
    nets = [
        ('MobileNetV2_1.0', load_module('NOS_MobileNetV2_1.0')),
        ('TSM_MobileNetV2_1.0', load_module('TSM_MobileNetV2_1.0')),
    ]

    mobilenet_shift_buffer = [torch.zeros(1, 12, 14, 14), torch.zeros(1, 20, 7, 7), torch.zeros(1, 20, 7, 7)]

    inputs = [(torch.rand(1, 3, 224, 224), *mobilenet_shift_buffer), ]

    device2host_target = {
        'pi': 'llvm -mcpu=cortex-a72 -target=armv7l-linux-gnueabihf',
        'tx2': 'llvm -mcpu=cortex-a57 -target=aarch64-linux-gnu',
        'nano': 'llvm -mcpu=cortex-a57 -target=aarch64-linux-gnu',
        'pi3b': 'llvm -device=arm_cpu -model=bcm2837 -target=armv8a-linux-gnu -mattr=+neon'
    }

    ctarget = device2host_target[args.device]
    df = pd.DataFrame(columns=['Name', 'Batchsize', 'Resolution', 'Target Device', 'FrameRate(Hz)', 'Std(Hz)'])
    for name, net in nets:
        for input in inputs:
            for device, target, ctx in [('cpu', ctarget, tvm.cpu()), ('gpu', 'cuda', tvm.gpu())]:
                #            for device, target, ctx in [('gpu', 'cuda', tvm.gpu())]:
                if args.no_gpu and device == 'gpu':
                    continue
                if args.device == 'pi3b_':
                    latency = test_latency_rpc(net, input, target, ctarget, 'pi3b')
                else:
                    latency = test_latency(net, input, target, ctarget, ctx)
                with open('{}_{}_{}.txt'.format(args.device, device, name), 'w') as f:
                    f.write(",".join(str(v) for v in latency))
                latency = latency[-len(latency):]
                fps = [1000.0 / v * input[0].size()[0] for v in latency]
                df.loc[len(df)] = [name, input[0].size()[0], input[0].size()[2], device, np.mean(fps), np.std(fps)]
                print(df.to_string(index=False))


main()
