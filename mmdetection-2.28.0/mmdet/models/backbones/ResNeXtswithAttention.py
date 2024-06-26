import torch
from mmdet.models.backbones import ResNeXt
from ..cv_lib.attention.CoordAttention import CoordAtt
from ..cv_lib.attention.SEAttention import SEAttention
from mmdet.models.builder import BACKBONES
 

class ResNeXtWithAttention(ResNeXt):
    def __init__(self , **kwargs):
        super(ResNeXtWithAttention, self).__init__(**kwargs)
        if self.depth in (50, 101, 152):
            self.dims = (256, 512, 1024, 2048)
        else:
            raise Exception()
        self.attention1 = self.get_attention_module(self.dims[1])     
        self.attention2 = self.get_attention_module(self.dims[2])     
        self.attention3 = self.get_attention_module(self.dims[3])     
    
    def get_attention_module(self, dim):
        raise NotImplementedError()
    
    def forward(self, x):
        outs = super().forward(x)
        outs = list(outs)
        outs[1] = self.attention1(outs[1])
        outs[2] = self.attention2(outs[2])
        outs[3] = self.attention3(outs[3])    
        outs = tuple(outs)
        return outs
    
@BACKBONES.register_module()
class ResNeXtWithCoordAttention(ResNeXtWithAttention):
    def __init__(self , **kwargs):
        super(ResNeXtWithCoordAttention, self).__init__(**kwargs)
 
    def get_attention_module(self, dim):
        return CoordAtt(inp=dim, oup=dim, reduction=32)
    
@BACKBONES.register_module()
class ResNeXtWithSEAttention(ResNeXtWithAttention):
    def __init__(self , **kwargs):
        super(ResNeXtWithSEAttention, self).__init__(**kwargs)
 
    def get_attention_module(self, dim):
        return SEAttention(channel=dim, reduction=16)
 
 
if __name__ == "__main__":
    # model = ResNet(depth=18)
    # model = ResNet(depth=34)
    # model = ResNeXt(depth=50)
    # model = ResNet(depth=101)    
    # model = ResNet(depth=152)
    # model = ResNetWithCoordAttention(depth=18)
    model = ResNeXtWithSEAttention(depth=50)
    x = torch.rand(1, 3, 224, 224)
    outs = model(x)
    # print(outs.shape)
    for i, out in enumerate(outs):
        print(i, out.shape)
