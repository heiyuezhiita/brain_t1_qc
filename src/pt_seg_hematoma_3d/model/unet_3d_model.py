# 2021.11.07 Unet 3d, input is 3d image, and multi-label (ICH and IVH)
from .unet_3d_modules import *


# Unet 3d model, using multi-layer
class Unet3dRaw(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='i', act_l='l'):
        super(Unet3dRaw, self).__init__()
        self.input_block = U3dDownBlock(in_ch=in_ch, out_ch=32, pool_l=None, norm_l=norm_l, act_l=act_l)

        self.down_block1 = U3dDownBlock(in_ch=32, out_ch=64, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block2 = U3dDownBlock(in_ch=64, out_ch=128, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block3 = U3dDownBlock(in_ch=128, out_ch=256, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block4 = U3dDownBlock(in_ch=256, out_ch=320, pool_l=pool_l, norm_l=norm_l, act_l=act_l)

        # out_ch must same as down_block3 channel
        self.up_block1 = U3dUpBlock(in_ch=320, out_ch=256, norm_l=norm_l, act_l=act_l)
        self.up_block2 = U3dUpBlock(in_ch=256, out_ch=128, norm_l=norm_l, act_l=act_l)
        self.up_block3 = U3dUpBlock(in_ch=128, out_ch=64, norm_l=norm_l, act_l=act_l)
        self.up_block4 = U3dUpBlock(in_ch=64, out_ch=32, norm_l=norm_l, act_l=act_l)

        self.out_conv = nn.Conv3d(in_channels=32, out_channels=out_ch, kernel_size=1, padding=0, stride=1)
        # self.out_conv = nn.Conv3d(in_channels=32, out_channels=out_ch, kernel_size=3, padding=1, stride=1)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        x1 = self.input_block(x)
        x_d2 = self.down_block1(x1)
        x_d3 = self.down_block2(x_d2)
        x_d4 = self.down_block3(x_d3)
        x_d5 = self.down_block4(x_d4)

        x_up1 = self.up_block1(x_d5, x_d4)
        x_up2 = self.up_block2(x_up1, x_d3)
        x_up3 = self.up_block3(x_up2, x_d2)
        x_up4 = self.up_block4(x_up3, x1)

        out = self.out_conv(x_up4)
        # out = self.softmax(x_out)

        return out


# Unet 3d model, using deep super vision, which includes Unet3dRaw
class Unet3dDSv(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='i', act_l='l', ct_out_bias=True,
                 is_deep_supervision=True):
        # ct_out_bias is ConvTranspose3d and out Conv bias
        super(Unet3dDSv, self).__init__()
        fn = [32, 64, 128, 256, 320]  # filter number

        self.is_deep_supervision = is_deep_supervision

        # down sample conv
        self.input_block = U3dDownBlock(in_ch=in_ch, out_ch=fn[0], pool_l=None, norm_l=norm_l, act_l=act_l)

        self.down_block1 = U3dDownBlock(in_ch=fn[0], out_ch=fn[1], pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block2 = U3dDownBlock(in_ch=fn[1], out_ch=fn[2], pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block3 = U3dDownBlock(in_ch=fn[2], out_ch=fn[3], pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block4 = U3dDownBlock(in_ch=fn[3], out_ch=fn[4], pool_l=pool_l, norm_l=norm_l, act_l=act_l)

        # out_ch must same as down_block3 channel
        self.up_block1 = U3dUpBlock(in_ch=fn[4], out_ch=fn[3], norm_l=norm_l, act_l=act_l, ct_out_bias=ct_out_bias)
        self.up_block2 = U3dUpBlock(in_ch=fn[3], out_ch=fn[2], norm_l=norm_l, act_l=act_l, ct_out_bias=ct_out_bias)
        self.up_block3 = U3dUpBlock(in_ch=fn[2], out_ch=fn[1], norm_l=norm_l, act_l=act_l, ct_out_bias=ct_out_bias)
        self.up_block4 = U3dUpBlock(in_ch=fn[1], out_ch=fn[0], norm_l=norm_l, act_l=act_l, ct_out_bias=ct_out_bias)

        # out
        if self.is_deep_supervision:
            self.conv_out1 = nn.Conv3d(fn[3], out_ch, kernel_size=1, bias=ct_out_bias)
            self.conv_out2 = nn.Conv3d(fn[2], out_ch, kernel_size=1, bias=ct_out_bias)
            self.conv_out3 = nn.Conv3d(fn[1], out_ch, kernel_size=1, bias=ct_out_bias)

        self.conv_out4 = nn.Conv3d(in_channels=fn[0], out_channels=out_ch, kernel_size=1, bias=ct_out_bias)

        # self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        x1 = self.input_block(x)
        x_d2 = self.down_block1(x1)
        x_d3 = self.down_block2(x_d2)
        x_d4 = self.down_block3(x_d3)
        x_d5 = self.down_block4(x_d4)

        x_up1 = self.up_block1(x_d5, x_d4)
        x_up2 = self.up_block2(x_up1, x_d3)
        x_up3 = self.up_block3(x_up2, x_d2)
        x_up4 = self.up_block4(x_up3, x1)

        if self.is_deep_supervision:
            out1 = self.conv_out1(x_up1)
            out2 = self.conv_out2(x_up2)
            out3 = self.conv_out3(x_up3)
            out4 = self.conv_out4(x_up4)

            out = [out4, out3, out2, out1]  # out4 is true out, using this order to match deep supervision weight
        else:
            out = self.conv_out4(x_up4)

        return out


# Unet 3d model with 3 down sample blocks, using multi-layer
class Unet3dL3(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='i', act_l='l'):
        super(Unet3dL3, self).__init__()
        self.input_block = U3dDownBlock(in_ch=in_ch, out_ch=32, pool_l=None, norm_l=norm_l, act_l=act_l)

        self.down_block1 = U3dDownBlock(in_ch=32, out_ch=64, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block2 = U3dDownBlock(in_ch=64, out_ch=128, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block3 = U3dDownBlock(in_ch=128, out_ch=256, pool_l=pool_l, norm_l=norm_l, act_l=act_l)

        # out_ch must same as down_block3 channel
        self.up_block1 = U3dUpBlock(in_ch=256, out_ch=128, norm_l=norm_l, act_l=act_l)
        self.up_block2 = U3dUpBlock(in_ch=128, out_ch=64, norm_l=norm_l, act_l=act_l)
        self.up_block3 = U3dUpBlock(in_ch=64, out_ch=32, norm_l=norm_l, act_l=act_l)

        self.out_conv = nn.Conv3d(in_channels=32, out_channels=out_ch, kernel_size=1, padding=0, stride=1)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        x1 = self.input_block(x)
        x_d2 = self.down_block1(x1)
        x_d3 = self.down_block2(x_d2)
        x_d4 = self.down_block3(x_d3)

        x_up1 = self.up_block1(x_d4, x_d3)
        x_up2 = self.up_block2(x_up1, x_d2)
        x_up3 = self.up_block3(x_up2, x1)

        out = self.out_conv(x_up3)
        # out = self.softmax(x_out)

        return out


# Unet Attention 3d model
class Unet3dAttention(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='i', act_l='l'):
        super(Unet3dAttention, self).__init__()
        self.input_block = U3dDownBlock(in_ch=in_ch, out_ch=32, pool_l=None, norm_l=norm_l, act_l=act_l)

        self.down_block1 = U3dDownBlock(in_ch=32, out_ch=64, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block2 = U3dDownBlock(in_ch=64, out_ch=128, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block3 = U3dDownBlock(in_ch=128, out_ch=256, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block4 = U3dDownBlock(in_ch=256, out_ch=320, pool_l=pool_l, norm_l=norm_l, act_l=act_l)

        # out_ch must same as down_block3 channel (i.e., in_sh_ch must same as out_ch)
        self.up_block1 = U3dUpAttentionBlock(in_up_ch=320, in_sh_ch=256, out_ch=256, norm_l=norm_l, act_l=act_l)
        self.up_block2 = U3dUpAttentionBlock(in_up_ch=256, in_sh_ch=128, out_ch=128, norm_l=norm_l, act_l=act_l)
        self.up_block3 = U3dUpAttentionBlock(in_up_ch=128, in_sh_ch=64, out_ch=64, norm_l=norm_l, act_l=act_l)
        self.up_block4 = U3dUpAttentionBlock(in_up_ch=64, in_sh_ch=32, out_ch=32, norm_l=norm_l, act_l=act_l)

        self.out_conv = nn.Conv3d(in_channels=32, out_channels=out_ch, kernel_size=1, padding=0, stride=1)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        x1 = self.input_block(x)
        x_d2 = self.down_block1(x1)
        x_d3 = self.down_block2(x_d2)
        x_d4 = self.down_block3(x_d3)
        x_d5 = self.down_block4(x_d4)

        x_up1 = self.up_block1(x_d5, x_d4)
        x_up2 = self.up_block2(x_up1, x_d3)
        x_up3 = self.up_block3(x_up2, x_d2)
        x_up4 = self.up_block4(x_up3, x1)

        out = self.out_conv(x_up4)
        # out = self.softmax(x_out)

        return out


# Res-Unet 3d model
class ResUnet3d(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='i', act_l='l'):
        super(ResUnet3d, self).__init__()
        self.conv1 = U3dResBlock(in_ch, 32, norm_l=norm_l, act_l=act_l)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = U3dResBlock(32, 64, norm_l=norm_l, act_l=act_l)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = U3dResBlock(64, 128, norm_l=norm_l, act_l=act_l)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = U3dResBlock(128, 256, norm_l=norm_l, act_l=act_l)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = U3dResBlock(256, 320, norm_l=norm_l, act_l=act_l)

        # up
        self.up6 = nn.ConvTranspose3d(320, 256, 2, stride=2)
        self.conv6 = U3dResBlock(512, 256, norm_l=norm_l, act_l=act_l)

        self.up7 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv7 = U3dResBlock(256, 128, norm_l=norm_l, act_l=act_l)

        self.up8 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv8 = U3dResBlock(128, 64, norm_l=norm_l, act_l=act_l)

        self.up9 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv9 = U3dResBlock(64, 32, norm_l=norm_l, act_l=act_l)

        self.conv10 = nn.Conv3d(32, out_ch, 1)
        # self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        # up
        up_6 = self.up6(c5)
        merge6 = torch.cat((up_6, c4), dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat((up_7, c3), dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat((up_8, c2), dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat((up_9, c1), dim=1)
        c9 = self.conv9(merge9)

        out = self.conv10(c9)

        # out = self.sf(c10)  # softmax for each channel (dim=1)
        return out


# Res-Unet 3d model, with deep supervision, can replace ResUnet3d
class ResUnet3dDSv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='i', act_l='l', is_deep_supervision=True):
        super(ResUnet3dDSv, self).__init__()
        fn = [32, 64, 128, 256, 320]  # filter number

        self.is_deep_supervision = is_deep_supervision

        # down
        self.conv1 = U3dResBlock(in_ch, fn[0], norm_l=norm_l, act_l=act_l)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = U3dResBlock(fn[0], fn[1], norm_l=norm_l, act_l=act_l)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = U3dResBlock(fn[1], fn[2], norm_l=norm_l, act_l=act_l)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = U3dResBlock(fn[2], fn[3], norm_l=norm_l, act_l=act_l)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = U3dResBlock(fn[3], fn[4], norm_l=norm_l, act_l=act_l)

        # up
        self.up6 = nn.ConvTranspose3d(fn[4], fn[3], 2, stride=2)
        self.conv6 = U3dResBlock(fn[3]*2, fn[3], norm_l=norm_l, act_l=act_l)

        self.up7 = nn.ConvTranspose3d(fn[3], fn[2], 2, stride=2)
        self.conv7 = U3dResBlock(fn[2]*2, fn[2], norm_l=norm_l, act_l=act_l)

        self.up8 = nn.ConvTranspose3d(fn[2], fn[1], 2, stride=2)
        self.conv8 = U3dResBlock(fn[1]*2, fn[1], norm_l=norm_l, act_l=act_l)

        self.up9 = nn.ConvTranspose3d(fn[1], fn[0], 2, stride=2)
        self.conv9 = U3dResBlock(fn[0]*2, fn[0], norm_l=norm_l, act_l=act_l)

        # out
        if self.is_deep_supervision:
            self.conv_out1 = nn.Conv3d(fn[3], out_ch, kernel_size=1)
            self.conv_out2 = nn.Conv3d(fn[2], out_ch, kernel_size=1)
            self.conv_out3 = nn.Conv3d(fn[1], out_ch, kernel_size=1)

        self.conv_out4 = nn.Conv3d(in_channels=fn[0], out_channels=out_ch, kernel_size=1, padding=0, stride=1)

        # self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        # up
        up_6 = self.up6(c5)
        merge6 = torch.cat((up_6, c4), dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat((up_7, c3), dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat((up_8, c2), dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat((up_9, c1), dim=1)
        c9 = self.conv9(merge9)

        if self.is_deep_supervision:
            out1 = self.conv_out1(c6)
            out2 = self.conv_out2(c7)
            out3 = self.conv_out3(c8)
            out4 = self.conv_out4(c9)

            out = [out4, out3, out2, out1]  # out4 is true out, using this order to match deep supervision weight
        else:
            out = self.conv_out4(c9)

        return out


# Res-Unet 3d model
class R2Unet3d(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, t=2, norm_l='i', act_l='l'):
        super(R2Unet3d, self).__init__()
        self.conv1 = R2U3dBlock(in_ch, 32, t=t, norm_l=norm_l, act_l=act_l)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = R2U3dBlock(32, 64, t=t, norm_l=norm_l, act_l=act_l)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = R2U3dBlock(64, 128, t=t, norm_l=norm_l, act_l=act_l)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = R2U3dBlock(128, 256, t=t, norm_l=norm_l, act_l=act_l)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = R2U3dBlock(256, 320, t=t, norm_l=norm_l, act_l=act_l)

        # up
        self.up6 = nn.ConvTranspose3d(320, 256, 2, stride=2)
        self.conv6 = R2U3dBlock(512, 256, t=t, norm_l=norm_l, act_l=act_l)

        self.up7 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv7 = R2U3dBlock(256, 128, t=t, norm_l=norm_l, act_l=act_l)

        self.up8 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv8 = R2U3dBlock(128, 64, t=t, norm_l=norm_l, act_l=act_l)

        self.up9 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv9 = R2U3dBlock(64, 32, t=t, norm_l=norm_l, act_l=act_l)

        self.conv10 = nn.Conv3d(32, out_ch, 1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        # up
        up_6 = self.up6(c5)
        merge6 = torch.cat((up_6, c4), dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat((up_7, c3), dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat((up_8, c2), dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat((up_9, c1), dim=1)
        c9 = self.conv9(merge9)

        out = self.conv10(c9)

        # out = nn.Softmax(dim=1)(c10)  # softmax for each channel (dim=1)
        return out


# Vnet 3d
class Vnet3d(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, norm_l='b', act_l='p'):
        super(Vnet3d, self).__init__()
        self.input_conv = VnetInputConv(in_ch=in_ch, out_ch=16, norm_l=norm_l, act_l=act_l)

        # n_conv not contains the down/up conv
        self.down_conv1 = VnetDownConv(in_ch=16, out_ch=32, n_conv=2, norm_l=norm_l, act_l=act_l)
        self.down_conv2 = VnetDownConv(in_ch=32, out_ch=64, n_conv=3, norm_l=norm_l, act_l=act_l)
        self.down_conv3 = VnetDownConv(in_ch=64, out_ch=128, n_conv=3, norm_l=norm_l, act_l=act_l)
        self.down_conv4 = VnetDownConv(in_ch=128, out_ch=256, n_conv=3, norm_l=norm_l, act_l=act_l)

        self.up_conv1 = VnetUpConv(in_ch=256, out_ch=256, n_conv=3, norm_l=norm_l, act_l=act_l)
        self.up_conv2 = VnetUpConv(in_ch=256, out_ch=128, n_conv=3, norm_l=norm_l, act_l=act_l)
        self.up_conv3 = VnetUpConv(in_ch=128, out_ch=64, n_conv=2, norm_l=norm_l, act_l=act_l)
        self.up_conv4 = VnetUpConv(in_ch=64, out_ch=32, n_conv=1, norm_l=norm_l, act_l=act_l)

        # out block
        self.out_conv = VnetOutputConv(in_ch=32, out_ch=out_ch, norm_l=norm_l, act_l=act_l)

    def forward(self, x):
        # shape of input x is B*C*D1*D2*D3
        # input block
        x1 = self.input_conv(x)

        # down sample block
        x_down1 = self.down_conv1(x1)
        x_down2 = self.down_conv2(x_down1)
        x_down3 = self.down_conv3(x_down2)
        x_down4 = self.down_conv4(x_down3)

        # up block
        x_up1 = self.up_conv1(x_down4, x_down3)
        x_up2 = self.up_conv2(x_up1, x_down2)
        x_up3 = self.up_conv3(x_up2, x_down1)
        x_up4 = self.up_conv4(x_up3, x1)

        # out block
        out = self.out_conv(x_up4)

        return out


# UnetSE 3d
class Unet3dSE(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='i', act_l='l', se_reduction=16):
        super(Unet3dSE, self).__init__()
        self.input_block = U3dSEDownBlock(in_ch=in_ch, out_ch=32, is_pool=False, norm_l=norm_l, act_l=act_l,
                                          se_reduction=se_reduction)

        self.down_block1 = U3dSEDownBlock(in_ch=32, out_ch=64, is_pool=True, norm_l=norm_l, act_l=act_l,
                                          se_reduction=se_reduction)
        self.down_block2 = U3dSEDownBlock(in_ch=64, out_ch=128, is_pool=True, norm_l=norm_l, act_l=act_l,
                                          se_reduction=se_reduction)
        self.down_block3 = U3dSEDownBlock(in_ch=128, out_ch=256, is_pool=True, norm_l=norm_l, act_l=act_l,
                                          se_reduction=se_reduction)
        self.down_block4 = U3dSEDownBlock(in_ch=256, out_ch=320, is_pool=True, norm_l=norm_l, act_l=act_l,
                                          se_reduction=se_reduction)

        # out_ch must same as down_block3 channel
        self.up_block1 = U3dSEUpBlock(in_ch=320, out_ch=256, norm_l=norm_l, act_l=act_l, se_reduction=se_reduction)
        self.up_block2 = U3dSEUpBlock(in_ch=256, out_ch=128, norm_l=norm_l, act_l=act_l, se_reduction=se_reduction)
        self.up_block3 = U3dSEUpBlock(in_ch=128, out_ch=64, norm_l=norm_l, act_l=act_l, se_reduction=se_reduction)
        self.up_block4 = U3dSEUpBlock(in_ch=64, out_ch=32, norm_l=norm_l, act_l=act_l, se_reduction=se_reduction)

        self.out_conv = nn.Conv3d(in_channels=32, out_channels=out_ch, kernel_size=1, padding=0, stride=1)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        x1 = self.input_block(x)
        x_d2 = self.down_block1(x1)
        x_d3 = self.down_block2(x_d2)
        x_d4 = self.down_block3(x_d3)
        x_d5 = self.down_block4(x_d4)

        x_up1 = self.up_block1(x_d5, x_d4)
        x_up2 = self.up_block2(x_up1, x_d3)
        x_up3 = self.up_block3(x_up2, x_d2)
        x_up4 = self.up_block4(x_up3, x1)

        out = self.out_conv(x_up4)
        # out = self.softmax(x_out)

        return out


# Unet++
class NestedUnet3d(nn.Module):
    def __init__(self, in_ch, out_ch, norm_l='i', act_l='l', is_deep_supervision=True):
        super(NestedUnet3d, self).__init__()
        filter_num = [32, 64, 128, 256, 320]

        # setting
        self.pool = nn.MaxPool3d(2)
        self.up = UpSample(scale_factor=2, mode='trilinear', align_corners=True)
        self.sf = nn.Softmax(dim=1)

        self.is_deep_supervision = is_deep_supervision

        # conv
        self.conv0_0 = Double3dConv(in_ch, filter_num[0], norm_l=norm_l, act_l=act_l)
        self.conv1_0 = Double3dConv(filter_num[0], filter_num[1], norm_l=norm_l, act_l=act_l)
        self.conv2_0 = Double3dConv(filter_num[1], filter_num[2], norm_l=norm_l, act_l=act_l)
        self.conv3_0 = Double3dConv(filter_num[2], filter_num[3], norm_l=norm_l, act_l=act_l)
        self.conv4_0 = Double3dConv(filter_num[3], filter_num[4], norm_l=norm_l, act_l=act_l)

        self.conv0_1 = Double3dConv(filter_num[0]+filter_num[1], filter_num[0], norm_l=norm_l, act_l=act_l)
        self.conv1_1 = Double3dConv(filter_num[1]+filter_num[2], filter_num[1], norm_l=norm_l, act_l=act_l)
        self.conv2_1 = Double3dConv(filter_num[2]+filter_num[3], filter_num[2], norm_l=norm_l, act_l=act_l)
        self.conv3_1 = Double3dConv(filter_num[3]+filter_num[4], filter_num[3], norm_l=norm_l, act_l=act_l)

        self.conv0_2 = Double3dConv(filter_num[0]*2+filter_num[1], filter_num[0], norm_l=norm_l, act_l=act_l)
        self.conv1_2 = Double3dConv(filter_num[1]*2+filter_num[2], filter_num[1], norm_l=norm_l, act_l=act_l)
        self.conv2_2 = Double3dConv(filter_num[2]*2+filter_num[3], filter_num[2], norm_l=norm_l, act_l=act_l)

        self.conv0_3 = Double3dConv(filter_num[0]*3+filter_num[1], filter_num[0], norm_l=norm_l, act_l=act_l)
        self.conv1_3 = Double3dConv(filter_num[1]*3+filter_num[2], filter_num[1], norm_l=norm_l, act_l=act_l)

        self.conv0_4 = Double3dConv(filter_num[0]*4+filter_num[1], filter_num[0], norm_l=norm_l, act_l=act_l)

        # out
        if self.is_deep_supervision:
            self.conv_out1 = nn.Conv3d(filter_num[0], out_ch, kernel_size=1)
            self.conv_out2 = nn.Conv3d(filter_num[0], out_ch, kernel_size=1)
            self.conv_out3 = nn.Conv3d(filter_num[0], out_ch, kernel_size=1)

        self.conv_out4 = nn.Conv3d(filter_num[0], out_ch, kernel_size=1)

    def forward(self, x):
        # input is B*C*D1*D2*D3
        # net 1
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        # net 2
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        # net 3
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # net 4
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.is_deep_supervision:
            out1 = self.conv_out1(x0_1)
            out2 = self.conv_out2(x0_2)
            out3 = self.conv_out3(x0_3)
            out4 = self.conv_out4(x0_4)

            out = [out4, out3, out2, out1]  # out4 is raw out, using this order to match deep supervision weight
        else:
            out = self.conv_out4(x0_4)

        return out


# Transformer Unet 3d
class Unet3dTransformer(nn.Module):
    def __init__(self, in_ch, out_ch, pool_l='m', norm_l='i', act_l='l'):
        super(Unet3dTransformer, self).__init__()
        self.input_block = U3dDownBlock(in_ch=in_ch, out_ch=32, pool_l=None, norm_l=norm_l, act_l=act_l)

        self.down_block1 = U3dDownBlock(in_ch=32, out_ch=64, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block2 = U3dDownBlock(in_ch=64, out_ch=128, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block3 = U3dDownBlock(in_ch=128, out_ch=256, pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block4 = U3dDownBlock(in_ch=256, out_ch=320, pool_l=pool_l, norm_l=norm_l, act_l=act_l)

        self.vit = ViT3d(img_shape=6*7*6, in_ch=320, embedding_dim=320, mlp_dim=512, head_num=4, block_num=8)

        # out_ch must same as down_block3 channel
        self.up_block1 = U3dUpBlock(in_ch=320, out_ch=256, norm_l=norm_l, act_l=act_l)
        self.up_block2 = U3dUpBlock(in_ch=256, out_ch=128, norm_l=norm_l, act_l=act_l)
        self.up_block3 = U3dUpBlock(in_ch=128, out_ch=64, norm_l=norm_l, act_l=act_l)
        self.up_block4 = U3dUpBlock(in_ch=64, out_ch=32, norm_l=norm_l, act_l=act_l)

        self.out_conv = nn.Conv3d(in_channels=32, out_channels=out_ch, kernel_size=1, padding=0, stride=1)
        # self.out_conv = nn.Conv3d(in_channels=32, out_channels=out_ch, kernel_size=3, padding=1, stride=1)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        x1 = self.input_block(x)
        x_d2 = self.down_block1(x1)
        x_d3 = self.down_block2(x_d2)
        x_d4 = self.down_block3(x_d3)
        x_d5 = self.down_block4(x_d4)

        x_trans = self.vit(x_d5)

        x_up1 = self.up_block1(x_trans, x_d4)
        x_up2 = self.up_block2(x_up1, x_d3)
        x_up3 = self.up_block3(x_up2, x_d2)
        x_up4 = self.up_block4(x_up3, x1)

        out = self.out_conv(x_up4)

        return out


# DR-Unet 3d
class DRUnet3d(nn.Module):
    def __init__(self, in_ch, out_ch, fn,
                 pool_l='m', norm_l='i', act_l='l',
                 is_norm_act_concate=False, is_deep_supervision=True):
        super(DRUnet3d, self).__init__()
        # fn is filter number list
        # fn = [32, 64, 128, 256, 320]  # load by arg

        self.is_deep_supervision = is_deep_supervision

        # down sample conv
        self.input_block = DRU3dDownBlock(in_ch=in_ch, out_ch=32, pool_l=None, norm_l=norm_l, act_l=act_l)

        self.down_block1 = DRU3dDownBlock(in_ch=fn[0], out_ch=fn[1], pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block2 = DRU3dDownBlock(in_ch=fn[1], out_ch=fn[2], pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block3 = DRU3dDownBlock(in_ch=fn[2], out_ch=fn[3], pool_l=pool_l, norm_l=norm_l, act_l=act_l)
        self.down_block4 = DRU3dDownBlock(in_ch=fn[3], out_ch=fn[4], pool_l=pool_l, norm_l=norm_l, act_l=act_l)

        # out_ch must same as down_block3 channel
        self.up_block1 = DRU3dUpBlock(in_ch=fn[4], out_ch=fn[3], norm_l=norm_l, act_l=act_l,
                                      is_norm_act_concate=is_norm_act_concate)
        self.up_block2 = DRU3dUpBlock(in_ch=fn[3], out_ch=fn[2], norm_l=norm_l, act_l=act_l,
                                      is_norm_act_concate=is_norm_act_concate)
        self.up_block3 = DRU3dUpBlock(in_ch=fn[2], out_ch=fn[1], norm_l=norm_l, act_l=act_l,
                                      is_norm_act_concate=is_norm_act_concate)
        self.up_block4 = DRU3dUpBlock(in_ch=fn[1], out_ch=fn[0], norm_l=norm_l, act_l=act_l,
                                      is_norm_act_concate=is_norm_act_concate)

        # out
        if self.is_deep_supervision:
            self.conv_out1 = nn.Conv3d(fn[3], out_ch, kernel_size=1)
            self.conv_out2 = nn.Conv3d(fn[2], out_ch, kernel_size=1)
            self.conv_out3 = nn.Conv3d(fn[1], out_ch, kernel_size=1)

        self.conv_out4 = nn.Conv3d(in_channels=fn[0], out_channels=out_ch, kernel_size=1, padding=0, stride=1)

        # self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        x1 = self.input_block(x)
        x_d2 = self.down_block1(x1)
        x_d3 = self.down_block2(x_d2)
        x_d4 = self.down_block3(x_d3)
        x_d5 = self.down_block4(x_d4)

        x_up1 = self.up_block1(x_d5, x_d4)
        x_up2 = self.up_block2(x_up1, x_d3)
        x_up3 = self.up_block3(x_up2, x_d2)
        x_up4 = self.up_block4(x_up3, x1)

        if self.is_deep_supervision:
            out1 = self.conv_out1(x_up1)
            out2 = self.conv_out2(x_up2)
            out3 = self.conv_out3(x_up3)
            out4 = self.conv_out4(x_up4)

            out = [out4, out3, out2, out1]  # out4 is true out, using this order to match deep supervision weight
        else:
            out = self.conv_out4(x_up4)

        return out


# MultiResUnet 3d
class MultiResUnet3d(nn.Module):
    def __init__(self, in_ch, out_ch, fn,
                 multi_res_alpha=1.67,
                 pool_l='m', norm_l='i', act_l='l',
                 is_norm_act_concate=False, is_deep_supervision=True, is_act_first=True):
        super(MultiResUnet3d, self).__init__()
        # fn is filter number list, input fn is number of filters in a corresponding UNet stage
        fn = [int(i * multi_res_alpha) for i in fn]  # using multi_res_alpha

        self.is_deep_supervision = is_deep_supervision

        # down sample conv
        self.input_block = MultiResU3dDownBlock(in_ch=in_ch, out_ch=fn[0], pool_l=None, norm_l=norm_l, act_l=act_l,
                                                is_act_first=is_act_first)

        self.down_block1 = MultiResU3dDownBlock(in_ch=fn[0], out_ch=fn[1], pool_l=pool_l, norm_l=norm_l, act_l=act_l,
                                                is_act_first=is_act_first)
        self.down_block2 = MultiResU3dDownBlock(in_ch=fn[1], out_ch=fn[2], pool_l=pool_l, norm_l=norm_l, act_l=act_l,
                                                is_act_first=is_act_first)
        self.down_block3 = MultiResU3dDownBlock(in_ch=fn[2], out_ch=fn[3], pool_l=pool_l, norm_l=norm_l, act_l=act_l,
                                                is_act_first=is_act_first)
        self.down_block4 = MultiResU3dDownBlock(in_ch=fn[3], out_ch=fn[4], pool_l=pool_l, norm_l=norm_l, act_l=act_l,
                                                is_act_first=is_act_first)

        # out_ch must same as down_block3 channel
        self.up_block1 = MultiResU3dUpBlock(in_ch=fn[4], out_ch=fn[3], res_path_length=1,
                                            norm_l=norm_l, act_l=act_l, is_norm_act_concate=is_norm_act_concate,
                                            is_act_first=is_act_first)
        self.up_block2 = MultiResU3dUpBlock(in_ch=fn[3], out_ch=fn[2], res_path_length=2,
                                            norm_l=norm_l, act_l=act_l, is_norm_act_concate=is_norm_act_concate,
                                            is_act_first=is_act_first)
        self.up_block3 = MultiResU3dUpBlock(in_ch=fn[2], out_ch=fn[1], res_path_length=3,
                                            norm_l=norm_l, act_l=act_l, is_norm_act_concate=is_norm_act_concate,
                                            is_act_first=is_act_first)
        self.up_block4 = MultiResU3dUpBlock(in_ch=fn[1], out_ch=fn[0], res_path_length=4,
                                            norm_l=norm_l, act_l=act_l, is_norm_act_concate=is_norm_act_concate,
                                            is_act_first=is_act_first)

        # out
        if self.is_deep_supervision:
            self.conv_out1 = nn.Conv3d(fn[3], out_ch, kernel_size=1, bias=False)
            self.conv_out2 = nn.Conv3d(fn[2], out_ch, kernel_size=1, bias=False)
            self.conv_out3 = nn.Conv3d(fn[1], out_ch, kernel_size=1, bias=False)

        self.conv_out4 = nn.Conv3d(in_channels=fn[0], out_channels=out_ch,
                                   kernel_size=1, padding=0, stride=1, bias=False)

        # self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        # input is B*C*W*H
        # down
        x1 = self.input_block(x)
        x_d2 = self.down_block1(x1)
        x_d3 = self.down_block2(x_d2)
        x_d4 = self.down_block3(x_d3)
        x_d5 = self.down_block4(x_d4)

        x_up1 = self.up_block1(x_d5, x_d4)
        x_up2 = self.up_block2(x_up1, x_d3)
        x_up3 = self.up_block3(x_up2, x_d2)
        x_up4 = self.up_block4(x_up3, x1)

        if self.is_deep_supervision:
            out1 = self.conv_out1(x_up1)
            out2 = self.conv_out2(x_up2)
            out3 = self.conv_out3(x_up3)
            out4 = self.conv_out4(x_up4)

            out = [out4, out3, out2, out1]  # out4 is true out, using this order to match deep supervision weight
        else:
            out = self.conv_out4(x_up4)

        return out


# U3d, but have more input args
class Unet3dBase(nn.Module):
    def __init__(self, in_ch, out_ch, fn,
                 pool_l='m', up_l='t', norm_l='i', act_l='l', drop_l='',
                 encode_conv_num=2, decode_conv_num=2,
                 pool_kwargs=None, conv_kwargs=None, norm_kwargs=None, act_kwargs=None, drop_kwargs=None,
                 is_deep_supervision=True):
        """
        Base Unet3d
        Args:
            in_ch:
            out_ch:
            fn: feature number list, contains feature number in each block
            pool_l:
            norm_l:
            act_l:
            drop_l:
            encode_conv_num:
            decode_conv_num:
            pool_kwargs: can input dict or list. if input list, must contains all args dict
            conv_kwargs: can input dict or list. if input list, must contains all args dict
            norm_kwargs:
            act_kwargs:
            drop_kwargs:
            is_dsv: is deep supervision
        """
        super(Unet3dBase, self).__init__()
        self.is_deep_supervision = is_deep_supervision

        # convert args dict to list
        pool_kwargs_tup = self._convert_args_to_tuple(pool_kwargs, len(fn))
        conv_kwargs_tup = self._convert_args_to_tuple(conv_kwargs, len(fn))
        norm_kwargs_tup = self._convert_args_to_tuple(norm_kwargs, len(fn))
        act_kwargs_tup = self._convert_args_to_tuple(act_kwargs, len(fn))
        drop_kwargs_tup = self._convert_args_to_tuple(drop_kwargs, len(fn))
        encode_conv_num_tup = self._convert_args_to_tuple(encode_conv_num, len(fn))
        decode_conv_num_tup = self._convert_args_to_tuple(decode_conv_num, len(fn)-1)

        # input blocks
        self.input_block = U3dEncodeBlock(
            in_ch, fn[0], pool_l=None, norm_l=norm_l, act_l=act_l, drop_l=drop_l,
            conv_num=encode_conv_num_tup[0], pool_kwargs=pool_kwargs_tup[0], conv_kwargs=conv_kwargs_tup[0],
            norm_kwargs=norm_kwargs_tup[0], act_kwargs=act_kwargs_tup[0], drop_kwargs=drop_kwargs_tup[0])

        # encode and decode blocks
        self.encode_list = []
        self.decode_list = []
        self.dsv_out_list = []
        for i in range(len(fn) - 1):
            # The (i+1)th encoding block corresponds to the (i)th decoding block
            # index 0 is input blocks, so using +1
            pool_kwargs = pool_kwargs_tup[i+1]
            conv_kwargs = conv_kwargs_tup[i+1]
            norm_kwargs = norm_kwargs_tup[i+1]
            act_kwargs = act_kwargs_tup[i+1]
            drop_kwargs = drop_kwargs_tup[i+1]
            encode_conv_num = encode_conv_num_tup[i+1]
            decode_conv_num = decode_conv_num_tup[i]

            self.encode_list.append(U3dEncodeBlock(
                fn[i], fn[i+1], pool_l=pool_l, norm_l=norm_l, act_l=act_l, drop_l=drop_l,
                conv_num=encode_conv_num, pool_kwargs=pool_kwargs, conv_kwargs=conv_kwargs,
                norm_kwargs=norm_kwargs, act_kwargs=act_kwargs, drop_kwargs=drop_kwargs))
            self.decode_list.append(U3dDecodeBlock(
                fn[i+1], fn[i], fn[i], up_l=up_l, norm_l=norm_l, act_l=act_l, drop_l=drop_l,
                conv_num=decode_conv_num, up_pool_kwargs=pool_kwargs, conv_kwargs=conv_kwargs,
                norm_kwargs=norm_kwargs, act_kwargs=act_kwargs, drop_kwargs=drop_kwargs))
            self.dsv_out_list.append(nn.Conv3d(fn[i], out_ch, kernel_size=1, padding=0, stride=1))

        self.encode_list = nn.ModuleList(self.encode_list)
        self.decode_list = nn.ModuleList(self.decode_list)
        self.dsv_out_list = nn.ModuleList(self.dsv_out_list)

    def forward(self, x):
        # input is B*C*D1*D2*D3
        x_pass_list = []
        out_list = []

        # Encode
        x = self.input_block(x)
        x_pass_list.append(x)
        for i in range(len(self.encode_list)):
            x = self.encode_list[i](x)
            if i+1 < len(self.encode_list):  # last encode block has not skip
                x_pass_list.append(x)

        # Decode
        for i in range(len(self.decode_list)):
            x = self.decode_list[-(i+1)](x, x_pass_list[-(i+1)])
            out_list.append(self.dsv_out_list[-(i+1)](x))

        if self.is_deep_supervision:
            out_list.reverse()  # convert actual output to first
            return out_list
        else:
            return out_list[-1]

    @ staticmethod
    def _convert_args_to_tuple(input_args, repeat_number):
        # convert inpu args to tuple.
        # list will be convert to tuple directly, and if not list will be repeat n times and as a tuple
        if isinstance(input_args, list):
            return tuple(input_args)
        elif not isinstance(input_args, tuple):
            return tuple([deepcopy(input_args) for _ in range(repeat_number)])
        else:
            return input_args
