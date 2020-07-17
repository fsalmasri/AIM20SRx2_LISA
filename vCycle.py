from modules import *


class Cyclic(nn.Module):
    def __init__(self):
        super(Cyclic, self).__init__()

        nChannel = 3
        nFeat2 = 48
        nFeat = 64
        
        self.loops = 10

        self.anal0 = Analysis(nChannel, nFeat, k= 3, str = 1)
        self.anal2 = Analysis(nChannel, nFeat2, k = 3, str= 1)
        self.synth2 = Sysnthesis(nChannel, nFeat2, 3)

        self.blocks = nn.ModuleList()
        for i in range(self.loops):
            self.blocks.append(block2(nFeat, nFeat2, 3))


    def forward(self, x):

        x = ((x * 2) - 1)

        x2 = nn.functional.interpolate(x, scale_factor= 2, mode='bicubic', align_corners=False)

        a = self.anal0(x)
        a0 = a
        a2 = self.anal2(x2)

        for i in range(self.loops):
            a0, a2, fs = self.blocks[i](a0, a2)

        SR0 = (self.synth2(a2) + 1) / 2

        return SR0

