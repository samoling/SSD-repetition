
#   该部分代码用于看网络结构
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.ssd import SSD300

if __name__ == "__main__":
    input_shape = [300, 300]
    num_classes = 21
    backbone    = "vgg"
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = SSD300(num_classes, backbone).to(device)
    summary(m, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(m.to(device), (dummy_input, ), verbose=False)

    # 选择乘2，参考YOLOX。
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))

