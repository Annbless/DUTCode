import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .pwcNet import PwcNet

import math
import pdb


class UNet1(nn.Module):
    def __init__(self):
        super(UNet1, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
                super(Encoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.LeakyReLU(0.2)
                )
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class Decoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, in_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0)
                )

                if tanh:
                    self.activ = nn.Tanh()
                else:
                    self.activ = nn.LeakyReLU(0.2)

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, in_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s * self.GateConv(x)

        self.enc0 = Encoder(6, 8, stride=1)
        self.enc1 = Encoder(8, 16, stride=2)
        self.enc2 = Encoder(16, 16, stride=2)
        self.enc3 = Encoder(16, 16, stride=2)

        self.dec0 = Decoder(16, 16, stride=1)
        # up-scaling + concat
        self.dec1 = Decoder(16+16, 16, stride=1)
        self.dec2 = Decoder(16+16, 16, stride=1)
        self.dec3 = Decoder(16+8, 16, stride=1)

        self.dec4 = Decoder(16, 3, stride=1, tanh=True)

    def forward(self, x1, x2):
        s0 = self.enc0(torch.cat([x1, x2], 1).cuda())
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        s4 = self.dec0(s3)
        # up-scaling + concat
        s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
        s5 = self.dec1(torch.cat([s4, s2], 1).cuda())
        s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
        s6 = self.dec2(torch.cat([s5, s1], 1).cuda())
        s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
        s7 = self.dec3(torch.cat([s6, s0], 1).cuda())

        out = self.dec4(s7)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        class ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super(ConvBlock, self).__init__()

                self.seq = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1,
                              stride=1, padding=0),
                    nn.LeakyReLU(0.2)
                )

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, in_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class ResBlock(nn.Module):
            def __init__(self, num_ch):
                super(ResBlock, self).__init__()

                self.seq = nn.Sequential(
                    nn.Conv2d(num_ch, num_ch, kernel_size=1,
                              stride=1, padding=0),
                    nn.LeakyReLU(0.2)
                )

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x) + x

        self.seq = nn.Sequential(
            ConvBlock(6, 32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ConvBlock(32, 3),
            nn.Tanh()
        )

    def forward(self, x1, x2):
        return self.seq(torch.cat([x1, x2], 1).cuda())


#############################################################################################################

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
                super(Encoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.ReLU()
                )
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class Decoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, in_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0)
                )

                if tanh:
                    self.activ = nn.Tanh()
                else:
                    self.activ = nn.ReLU()

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, in_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s * self.GateConv(x)

        self.enc0 = Encoder(16, 32, stride=1)
        self.enc1 = Encoder(32, 32, stride=2)
        self.enc2 = Encoder(32, 32, stride=2)
        self.enc3 = Encoder(32, 32, stride=2)

        self.dec0 = Decoder(32, 32, stride=1)
        # up-scaling + concat
        self.dec1 = Decoder(32+32, 32, stride=1)
        self.dec2 = Decoder(32+32, 32, stride=1)
        self.dec3 = Decoder(32+32, 32, stride=1)

        self.dec4 = Decoder(32, 3, stride=1, tanh=True)

    def forward(self, w1, w2, flo1, flo2, fr1, fr2):
        s0 = self.enc0(torch.cat([w1, w2, flo1, flo1, fr1, fr2], 1).cuda())
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        s4 = self.dec0(s3)
        # up-scaling + concat
        s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
        s5 = self.dec1(torch.cat([s4, s2], 1).cuda())
        s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
        s6 = self.dec2(torch.cat([s5, s1], 1).cuda())
        s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
        s7 = self.dec3(torch.cat([s6, s0], 1).cuda())

        out = self.dec4(s7)
        return out


class DIFNet2(nn.Module):
    def __init__(self):
        super(DIFNet2, self).__init__()

        class Backward(torch.nn.Module):
            def __init__(self):
                super(Backward, self).__init__()
            # end

            def forward(self, tensorInput, tensorFlow, scale=1.0):
                if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
                    self.tensorPartial = torch.FloatTensor().resize_(tensorFlow.size(
                        0), 1, tensorFlow.size(2), tensorFlow.size(3)).fill_(1.0).cuda()
                # end

                if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
                    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                        1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                        1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                    self.tensorGrid = torch.cat(
                        [tensorHorizontal, tensorVertical], 1).cuda()
                # end
                # pdb.set_trace()
                tensorInput = torch.cat([tensorInput, self.tensorPartial], 1)
                tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(
                    3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

                tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(
                    self.tensorGrid + tensorFlow*scale).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

                tensorMask = tensorOutput[:, -1:, :, :]
                tensorMask[tensorMask > 0.999] = 1.0
                tensorMask[tensorMask < 1.0] = 0.0

                return tensorOutput[:, :-1, :, :] * tensorMask

        # PWC
        self.pwc = PwcNet()
        self.pwc.load_state_dict(torch.load('./ckpt/sintel.pytorch'))
        self.pwc.eval()

        # Warping layer
        self.warpLayer = Backward()
        self.warpLayer.eval()

        # UNets
        self.UNet2 = UNet2()
        self.ResNet2 = ResNet2()

    def warpFrame(self, fr_1, fr_2, scale=1.0):
        with torch.no_grad():
            # Due to Pyramid method?
            temp_w = int(math.floor(math.ceil(fr_1.size(3) / 64.0) * 64.0))
            temp_h = int(math.floor(math.ceil(fr_1.size(2) / 64.0) * 64.0))

            temp_fr_1 = torch.nn.functional.interpolate(
                input=fr_1, size=(temp_h, temp_w), mode='nearest')
            temp_fr_2 = torch.nn.functional.interpolate(
                input=fr_2, size=(temp_h, temp_w), mode='nearest')

            flo = 20.0 * torch.nn.functional.interpolate(input=self.pwc(temp_fr_1, temp_fr_2), size=(
                fr_1.size(2), fr_1.size(3)), mode='bilinear', align_corners=False)
            return self.warpLayer(fr_2, flo, scale), flo

    def forward(self, fr1, fr2, f3, fs2, fs1, scale):
        w1, flo1 = self.warpFrame(fs2, fr1, scale=scale)
        w2, flo2 = self.warpFrame(fs1, fr2, scale=scale)

        I_int = self.UNet2(w1, w2, flo1, flo2, fr1, fr2)
        f_int, flo_int = self.warpFrame(I_int, f3)

        fhat = self.ResNet2(I_int, f_int, flo_int, f3)
        return fhat, I_int


class ResNet2(nn.Module):
    def __init__(self):
        super(ResNet2, self).__init__()

        class ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super(ConvBlock, self).__init__()

                self.seq = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1,
                              stride=1, padding=0),
                    nn.ReLU()
                )

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, in_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class ResBlock(nn.Module):
            def __init__(self, num_ch):
                super(ResBlock, self).__init__()

                self.seq = nn.Sequential(
                    nn.Conv2d(num_ch, num_ch, kernel_size=1,
                              stride=1, padding=0),
                    nn.ReLU()
                )

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x) + x

        self.seq = nn.Sequential(
            ConvBlock(11, 32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ConvBlock(32, 3),
            nn.Tanh()
        )

    def forward(self, I_int, f_int, flo_int, f3):
        return self.seq(torch.cat([I_int, f_int, flo_int, f3], 1).cuda())


#############################################################################################################
class UNetFlow(nn.Module):
    def __init__(self):
        super(UNetFlow, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
                super(Encoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.LeakyReLU(0.2)
                )
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class Decoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0)
                )

                if tanh:
                    self.activ = nn.Tanh()
                else:
                    self.activ = nn.LeakyReLU(0.2)

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s * self.GateConv(x)

        self.enc0 = Encoder(4, 32, stride=1)
        self.enc1 = Encoder(32, 32, stride=2)
        self.enc2 = Encoder(32, 32, stride=2)
        self.enc3 = Encoder(32, 32, stride=2)

        self.dec0 = Decoder(32, 32, stride=1)
        # up-scaling + concat
        self.dec1 = Decoder(32+32, 32, stride=1)
        self.dec2 = Decoder(32+32, 32, stride=1)
        self.dec3 = Decoder(32+32, 32, stride=1)

        self.dec4 = Decoder(32, 2, stride=1)

    def forward(self, x1, x2):
        s0 = self.enc0(torch.cat([x1, x2], 1).cuda())
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        s4 = self.dec0(s3)
        # up-scaling + concat
        s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
        s5 = self.dec1(torch.cat([s4, s2], 1).cuda())
        s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
        s6 = self.dec2(torch.cat([s5, s1], 1).cuda())
        s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
        s7 = self.dec3(torch.cat([s6, s0], 1).cuda())

        out = self.dec4(s7)
        return out


class UNet3(nn.Module):
    def __init__(self):
        super(UNet3, self).__init__()

        class Encoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
                super(Encoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.ReLU()
                )
                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class Decoder(nn.Module):
            def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
                super(Decoder, self).__init__()

                self.seq = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0)
                )

                if tanh:
                    self.activ = nn.Tanh()
                else:
                    self.activ = nn.ReLU()

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(in_nc, out_nc, kernel_size=k_size,
                              stride=stride, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                s = self.seq(x)
                s = self.activ(s)
                return s * self.GateConv(x)

        self.enc0 = Encoder(6, 32, stride=1)
        self.enc1 = Encoder(32, 32, stride=2)
        self.enc2 = Encoder(32, 32, stride=2)
        self.enc3 = Encoder(32, 32, stride=2)

        self.dec0 = Decoder(32, 32, stride=1)
        # up-scaling + concat
        self.dec1 = Decoder(32+32, 32, stride=1)
        self.dec2 = Decoder(32+32, 32, stride=1)
        self.dec3 = Decoder(32+32, 32, stride=1)

        self.dec4 = Decoder(32, 3, stride=1, tanh=True)

    def forward(self, w1, w2):
        s0 = self.enc0(torch.cat([w1, w2], 1).cuda())
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        s4 = self.dec0(s3)
        # up-scaling + concat
        s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
        s5 = self.dec1(torch.cat([s4, s2], 1).cuda())
        s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
        s6 = self.dec2(torch.cat([s5, s1], 1).cuda())
        s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
        s7 = self.dec3(torch.cat([s6, s0], 1).cuda())

        out = self.dec4(s7)
        return out


class DIFNet3(nn.Module):
    def __init__(self):
        super(DIFNet3, self).__init__()

        class Backward(torch.nn.Module):
            def __init__(self):
                super(Backward, self).__init__()
            # end

            def forward(self, tensorInput, tensorFlow, scale=1.0):
                if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
                    self.tensorPartial = torch.FloatTensor().resize_(tensorFlow.size(
                        0), 1, tensorFlow.size(2), tensorFlow.size(3)).fill_(1.0).cuda()
                # end

                if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
                    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                        1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
                    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                        1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

                    self.tensorGrid = torch.cat(
                        [tensorHorizontal, tensorVertical], 1).cuda()
                # end
                # pdb.set_trace()
                tensorInput = torch.cat([tensorInput, self.tensorPartial], 1)
                tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(
                    3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

                tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(
                    self.tensorGrid + tensorFlow*scale).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

                tensorMask = tensorOutput[:, -1:, :, :]
                tensorMask[tensorMask > 0.999] = 1.0
                tensorMask[tensorMask < 1.0] = 0.0

                return tensorOutput[:, :-1, :, :] * tensorMask

        # PWC
        self.pwc = PwcNet()
        self.pwc.load_state_dict(torch.load('./trained_models/sintel.pytorch'))
        self.pwc.eval()

        # Warping layer
        self.warpLayer = Backward()
        self.warpLayer.eval()

        # UNets
        self.UNetFlow = UNetFlow()
        self.UNet = UNet3()
        self.ResNet = ResNet3()

    def warpFrame(self, fr_1, fr_2, scale=1.0):
        with torch.no_grad():
            # Due to Pyramid method?
            temp_w = int(math.floor(math.ceil(fr_1.size(3) / 64.0) * 64.0))
            temp_h = int(math.floor(math.ceil(fr_1.size(2) / 64.0) * 64.0))

            temp_fr_1 = torch.nn.functional.interpolate(
                input=fr_1, size=(temp_h, temp_w), mode='nearest')
            temp_fr_2 = torch.nn.functional.interpolate(
                input=fr_2, size=(temp_h, temp_w), mode='nearest')

            flo = 20.0 * torch.nn.functional.interpolate(input=self.pwc(temp_fr_1, temp_fr_2), size=(
                fr_1.size(2), fr_1.size(3)), mode='bilinear', align_corners=False)
            return self.warpLayer(fr_2, flo, scale), flo

    def forward(self, fr1, fr2, f3, fs2, fs1, scale):
        _, flo1 = self.warpFrame(fs2, fr1, scale=scale)
        _, flo2 = self.warpFrame(fs1, fr2, scale=scale)

        # refine flow
        flo1_ = self.UNetFlow(flo1, flo2)
        flo2_ = self.UNetFlow(flo2, flo1)

        w1 = self.warpLayer(fr1, flo1_, scale)
        w2 = self.warpLayer(fr2, flo2_, scale)

        I_int = self.UNet(w1, w2)
        f_int, _ = self.warpFrame(I_int, f3)

        fhat = self.ResNet(I_int, f_int)
        return fhat, I_int


class ResNet3(nn.Module):
    def __init__(self):
        super(ResNet3, self).__init__()

        class ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super(ConvBlock, self).__init__()

                self.seq = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1,
                              stride=1, padding=0),
                    nn.ReLU()
                )

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, in_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x)

        class ResBlock(nn.Module):
            def __init__(self, num_ch):
                super(ResBlock, self).__init__()

                self.seq = nn.Sequential(
                    nn.Conv2d(num_ch, num_ch, kernel_size=1,
                              stride=1, padding=0),
                    nn.ReLU()
                )

                self.GateConv = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(num_ch, num_ch, kernel_size=3,
                              stride=1, padding=0),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.seq(x) * self.GateConv(x) + x

        self.seq = nn.Sequential(
            ConvBlock(6, 32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
            ConvBlock(32, 3),
            nn.Tanh()
        )

    def forward(self, I_int, f_int):
        return self.seq(torch.cat([I_int, f_int], 1).cuda())


############################ GAN Discriminator###############################

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, fr_2):
        #x = torch.cat((fr_1, fr_2), 1)
        x = self.seq(fr_2)
        x = F.avg_pool2d(x, x.size()[2:]).view(-1, x.size()[0]).squeeze()
        out = torch.sigmoid(x)
        return out
