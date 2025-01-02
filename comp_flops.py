# -- coding: utf-8 --
import torch
import torchvision
from thop import profile
from nets.ssd import SSD300

# Model
print('==> Building model..')
# model = torchvision.models.alexnet(pretrained=False)
model = SSD300(num_classes=1, backbone_name='mobilenetv2', pretrained=False)
dummy_input = torch.randn(1, 3, 896, 992)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0)) 