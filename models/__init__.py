#from .proxyless_nas import proxyless_gpu, proxyless_cpu, proxyless_mobile, proxyless_test, proxyless_tests, proxyless_from_config
from .mobilenetv2 import MobileNetV2
from .mobilenetv2_tsm import tsm_mobilenet_v2_140, tsm_mobilenet_v2_100, mobilenet_v2_100
from .resnet_tsm import resnet50, resnet50tsm
import os

__ALL__ = ['load_module', 'proxyless_from_config']

def load_module(name):
    if name == 'MobileNetV2_0.5':
        return MobileNetV2(width_mult=0.5)
    elif name == 'MobileNetV2_1.0':
        return MobileNetV2(width_mult=1.0)
    elif name == 'MobileNetV2_1.4':
        return MobileNetV2(width_mult=1.4)
#    elif name == 'Proxyless_GPU':
#        return proxyless_gpu(pretrained=False)
#    elif name == 'Proxyless_CPU':
#        return proxyless_cpu(pretrained=False)
#    elif name == 'Proxyless_Mobile':
#        return proxyless_mobile(pretrained=False)
#    elif name == 'Proxyless_Test':
#        return proxyless_test()
    elif name == 'TSM_MobileNetV2_1.4':
        return tsm_mobilenet_v2_140()
    elif name == 'TSM_MobileNetV2_1.0':
        return tsm_mobilenet_v2_100()
    elif name == 'NOS_MobileNetV2_1.0':
        return mobilenet_v2_100()
    elif name == 'TSM_ResNet50':
        return resnet50tsm(pretrained=False)
    elif name == 'ResNet50':
        return resnet50(pretrained=False)
    else:
        raise ValueError("Unknown model {name}".format(name=name))

def load_tflite_buf(name):
    cur_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    if name == 'EfficientNetB0-FP32':
         return open(os.path.join(cur_dir, 'EfficientNetB0-fp32.tflite'), 'rb').read()
    elif name == 'NasNet-Mobile-FP32':
        return open(os.path.join(cur_dir, 'nasnet-mobile-fp32.tflite'), 'rb').read()
    elif name == 'MobileNetV2_1.0':
        return open(os.path.join(cur_dir, 'MobileNetV2-fp32.tflite'), 'rb').read()
    elif name == 'FBNet-A':
        return open(os.path.join(cur_dir, 'FBNet-A.tflite'), 'rb').read()
    elif name == 'FBNet-B':
        return open(os.path.join(cur_dir, 'FBNet-B.tflite'), 'rb').read()
    elif name == 'FBNet-C':
        return open(os.path.join(cur_dir, 'FBNet-C.tflite'), 'rb').read()
    else:
        raise ValueError()

def load_test_modules():
    pass
#    return proxyless_tests()
