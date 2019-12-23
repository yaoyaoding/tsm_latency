import numpy as np
from typing import Tuple
import io
import tvm
import tvm.relay
import time
import cv2
import torch
import torchvision
import torch.onnx
from PIL import Image, ImageOps
import onnx
import tvm.contrib.graph_runtime as graph_runtime
from mobilenetv2_tsm import MobileNetV2


def torch2executor(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    torch_module.eval()
    input_names = []
    input_shapes = {}
    with torch.no_grad():
        for index, torch_input in enumerate(torch_inputs):
            name = str(index)
            input_names.append(name)
            input_shapes[name] = torch_input.shape
        buffer = io.BytesIO()
        torch.onnx.export(torch_module, torch_inputs, buffer, input_names=input_names)
        buffer.seek(0, 0)
        onnx_model = onnx.load_model(buffer)
        relay_module, params = tvm.relay.frontend.from_onnx(onnx_model, shape=input_shapes)
    with tvm.relay.build_config(opt_level=3):
        graph, tvm_module, params = tvm.relay.build(relay_module, target, params=params)
    ctx = tvm.gpu() if target.startswith('cuda') else tvm.cpu()
    graph_module = graph_runtime.create(graph, tvm_module, ctx)
    for pname, pvalue in params.items():
        graph_module.set_input(pname, pvalue)

    def executor(inputs: Tuple[np.ndarray]):
        for index, value in enumerate(inputs):
            graph_module.set_input(index, tvm.nd.array(value, ctx))
        graph_module.run()
        return tuple(graph_module.get_output(index) for index in range(len(inputs)))
    return executor


def get_executor():
    torch_module = MobileNetV2(use_shift=True, n_class=1000, input_size=224, width_mult=1.)
    torch_inputs = (torch.rand(1, 3, 224, 224),
                    torch.zeros(1, 12, 14, 14), torch.zeros(1, 20, 7, 7), torch.zeros(1, 20, 7, 7))
    return torch2executor(torch_module, torch_inputs, target='llvm')


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame

def get_transform():
    # Initialize frame transforms.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Scale(256, Image.BILINEAR),
        torchvision.transforms.CenterCrop(224)
    ])
    return transform


def main():
    cap = cv2.VideoCapture(0)

    t = None
    index = 0
    transform = get_transform()
    while True:
        _, frame = cap.read()   # (480, 640, 3) 0 ~ 255
        assert isinstance(frame, np.ndarray)
        frame = transform(frame)   #

#        frame = cv2.resize(frame, (224, 224))
#        print(type(frame))
#        print(frame.shape)
#        print(frame.min())
#        print(frame.max())
#        frame = transform(frame)
#        print(frame.shape)
#        exit(0)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if t is None:
            t = time.time()
        else:
            nt = time.time()
            print(f"{index} {1.0 / (nt - t):.2f} fps")
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()

main()
