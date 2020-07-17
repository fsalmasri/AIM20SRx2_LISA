import torch.nn as nn

def activF():
    return nn.PReLU()


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            activF(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Analysis(nn.Module):
    def __init__(self, nChannels, nFeat, k, str=1, p=0):
        super(Analysis, self).__init__()

        if not p:
            pad = (k - 1) // 2
        else:
            pad = p
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(nChannels, nFeat, k, stride=str, padding=0, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class Sysnthesis(nn.Module):
    def __init__(self, nChannels, nFeat, k, str=1):
        super(Sysnthesis, self).__init__()

        pad = (k - 1) // 2
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(nFeat, nChannels, k, stride=str, padding=0, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


def main_layer(nFeat):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(nFeat, nFeat, 3, stride=1, padding=0, bias=True, groups=1),
        activF(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(nFeat, nFeat, 3, stride=1, padding=0, bias=True, groups=1),
        activF()
    )


class upscal(nn.Module):
    def __init__(self, nFeat, nFeat2, layers=0):
        super(upscal, self).__init__()

        self.upper = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nFeat, nFeat2, kernel_size=3, stride=1, padding=0, bias=True, groups=1),
        )

        self.layers = layers

        self.mConv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=0, bias=True, groups=1),
            activF()
        )

    def forward(self, x):
        feat = self.mConv(x)
        u = self.upper(feat)
        return u


class downer(nn.Module):
    def __init__(self, nFeat, nFeat2, layers=0):
        super(downer, self).__init__()

        self.layers = layers

        self.DB = main_layer(nFeat2)
        self.ca = CALayer(nFeat, 12)

        self.down = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(nFeat2, nFeat, 5, stride=2, padding=0, bias=True, groups=1)
        )

    def forward(self, x):
        d1 = self.DB(x)
        d1 = self.down(d1)
        d1 = self.ca(d1)

        return d1


class block2(nn.Module):
    def __init__(self, nFeat, nFeat2, n):
        super(block2, self).__init__()

        self.nloops = n
        self.upscaler = upscal(nFeat, nFeat2)

        self.innerBlocks = nn.ModuleList()
        for i in range(self.nloops):
            self.innerBlocks.append(upscal(nFeat, nFeat2))
            self.innerBlocks.append(downer(nFeat, nFeat2, 0))

    def forward(self, a0, a2):

        a00, f00 = a0, a2
        for i in range(0, self.nloops, 2):
            u = self.innerBlocks[i](a00)
            f00 = f00 + u
            d = self.innerBlocks[i + 1](f00)
            a00 = a00 + d

        a0 = a0 + d

        u = self.upscaler(a0)
        a2 = u + a2

        return a0, a2, [f00]