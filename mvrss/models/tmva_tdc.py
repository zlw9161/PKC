# MVRSS-Net w/ temporal deformable Conv
# deformable Conv ref: https://github.com/4uiiurz1/pytorch-deform-conv-v2
# tmva_tdc: (3dconv_3x3x1)*2 + (reshape TxHxW -> T (HxW)s) + (tdc_3x3)*2 + temp gap/gmp;
#           maxpool 2d + ASPP; maxpool 2d + latent.
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Double3DConvBlock(nn.Module):
    """ (3D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, strd, pad, dil):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=k_size, stride=strd, padding=pad, dilation=dil),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

# vanilla dcn zlw @20220620
class DeformConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
            input: x in the shape of BxCxHxW 
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv block for deformed receptive field in x
        self.conv_df = nn.Conv2d(in_ch, out_ch,
                                 kernel_size=kernel_size,
                                 stride=kernel_size,
                                 bias=bias)
        # learnable offset conv block
        self.conv_offset = nn.Conv2d(in_ch, 2*kernel_size*kernel_size, 
                                     kernel_size=3,
                                     padding=1,
                                     stride=stride)
        # conv_offset weights initialization
        nn.init.constant_(self.conv_offset.weight, 0)
        # backward pass registration for conv_offset
        # self.conv_offset.register_backward_hook(self._set_lr)
        # switch for modulated dcn (dcn_v2)
        self.modulation = modulation
        if modulation:
            self.conv_mask = nn.Conv2d(in_ch, kernel_size*kernel_size,
                                       kernel_size=3,
                                       padding=1,
                                       stride=stride)
            nn.init.constant_(self.conv_mask.weight, 0)
            self.conv_mask.register_backward_hook(self._set_lr)

    @staticmethod
    # set learning rate for conv_offset and conv_mask
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))   

    def forward(self, x):
        # getting offset
        # (b, c, h, w) ===> (b, 2*kernel_size*kernel_size, h, w)
        offset = self.conv_offset(x)
        # getting mask
        if self.modulation:
            # (b, c, h, w) ===> (b, kernel_size*kernel_size, h, w)
            mask = torch.sigmoid(self.conv_mask(x))

        dtype = offset.data.type()
        k = self.kernel_size
        # number of offset points in one time sampling (k*k)
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # let p denote the deformed positions: p_df = p_o + \delta p + p_n
        # p shape: Bx2NxHxW
        p = self._get_p(offset, dtype)

        # memory contiguous for p, and put channel dimension at the last
        p = p.contiguous().permute(0, 2, 3, 1)
        # let bi-linear intp. operate on q = [q_lt, q_rb, q_lb, q_rt] for each p
        # left top q is the nearest integer number of p
        q_lt = p.detach().floor()
        # left top q + 1 = the q in right below
        q_rb = q_lt + 1

        # restrict q in the range of [(0, 0), (H-1, W-1)]
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        # restrict p in the range of [(0, 0), (H-1, W-1)]
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # using q to get bilinear kernel weights for p
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        # getting x(q) from input x
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, kernel_size*kernel_size)
        # weighted summation of bilinear kernel interpolation
        x_resamp = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation for dcn v2
        if self.modulation:
            # (b, kernel_size*kernel_size, h, w) ===> (b, h, w, kernel_size*kernel_size)
            m = m.contiguous().permute(0, 2, 3, 1)
            # (b, h, w, kernel_size*kernel_size) ===>  (b, 1, h, w, kernel_size*kernel_size)
            m = m.unsqueeze(dim=1)
            # (b, c, h, w, kernel_size*kernel_size)
            m = torch.cat([m for _ in range(x_resamp.size(1))], dim=1)
            x_resamp *= m
        # x_resamp: (b, c, h, w, kernel_size*kernel_size)
        # x_resamp: (b, c, h*kernel_size, w*kernel_size)
        x_resamp = self._reshape_x_resamp(x_resamp, k)
        # out: (b, c, h, w)
        out = self.conv_df(x_resamp)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    # getting p0 (the center point coord) from the input x
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # getting p (p = p0 + pn + \delta p) 
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    # getting x(q) from input x
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_q = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_q

    @staticmethod
    def _reshape_x_resamp(x_resamp, k):
        b, c, h, w, N = x_resamp.size()
        x_resamp = torch.cat([x_resamp[..., s:s+k].contiguous().view(b, c, h, w*k) for s in range(0, N, k)], dim=-1)
        x_resamp = x_resamp.contiguous().view(b, c, h*k, w*k)
        return x_resamp

# double TDConvBlock definition zlw @20220622
class DoubleTDConv2D(nn.Module):
    """ input (x): BxCxTxHxW """
    """ reshape x as BTxCxHxW """
    """ reshape output as BxCxTxHxW """ 
    """ (2D DeformConv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, strd, bool_mod):
        super(DoubleTDConv2D, self).__init__()
        self.df_conv1 = DeformConv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, stride=strd, modulation=bool_mod)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.df_conv2 = DeformConv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, stride=strd, modulation=bool_mod)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        b, c, t, h, w = x.size()
        # permute x as: BxTxCxHxW
        x = torch.permute(x, (0, 2, 1, 3, 4))
        # flatten x as BTxCxHxW
        x = torch.flatten(x, start_dim=0, end_dim=1)
        
        x = self.act1(self.bn1(self.df_conv1(x)))
        #print("x:", type(x))
        x = self.act2(self.bn2(self.df_conv2(x)))

        # reshape x as: BxCxTxHxW
        x = x.unsqueeze(dim=0)
        x = x.contiguous().view(b, c, t, h, w)
        return x

class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling
    Parallel conv blocks with different dilation rate
    """

    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.global_avg_pool = nn.AvgPool2d((64, 64))
        self.conv1_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, dilation=1)
        self.single_conv_block1_1x1 = ConvBlock(in_ch, out_ch, k_size=1, pad=0, dil=1)
        self.single_conv_block1_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=6, dil=6)
        self.single_conv_block2_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=12, dil=12)
        self.single_conv_block3_3x3 = ConvBlock(in_ch, out_ch, k_size=3, pad=18, dil=18)

    def forward(self, x):
        x1 = F.interpolate(self.global_avg_pool(x), size=(64, 64), align_corners=False,
                           mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x2 = self.single_conv_block1_1x1(x)
        x3 = self.single_conv_block1_3x3(x)
        x4 = self.single_conv_block2_3x3(x)
        x5 = self.single_conv_block3_3x3(x)
        x_cat = torch.cat((x2, x3, x4, x5, x1), 1)
        return x_cat

# adding temporal gap and tdc block, zlw @20220622
# replacing 2d pooling -> 3d pooling
class EncodingBranch(nn.Module):
    """
    Encoding branch for a single radar view

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view.
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    """

    def __init__(self, signal_type, frm_num):
        super().__init__()
        self.signal_type = signal_type
        # shared 3d conv as paralleled shared 2d conv
        self.double_3dconv_block1 = Double3DConvBlock(in_ch=1, out_ch=128,
                                                      k_size=(1, 3, 3),
                                                      strd=(1, 1, 1),
                                                      pad=(0, 1, 1),
                                                      dil=1)
        self.doppler_max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                             stride=(1, 2, 1))
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                     stride=(1, 2, 2))
        '''
        self.double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                  pad=1, dil=1)
        '''
        self.double_tdc_block = DoubleTDConv2D(in_ch=128, out_ch=128,
                                               k_size=3,
                                               pad=1,
                                               strd=1,
                                               bool_mod=False)
        self.temporal_gap = nn.AvgPool3d(kernel_size=(frm_num, 1, 1),
                                         stride=None,
                                         padding=0)
        self.single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1,
                                                pad=0, dil=1)

    def forward(self, x):
        x1 = self.double_3dconv_block1(x)
        #print("x1_3dconv: ", x1.size())
        # x1 = torch.squeeze(x1, 2)  # remove temporal dimension

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x1 = F.pad(x1, (0, 1, 0, 0), "constant", 0)
            x1 = self.doppler_max_pool(x1)
        else:
            x1 = self.max_pool(x1)

        #print("x1_down: ", x1.size())
        # x2 = self.double_conv_block2(x1_down)
        x2 = self.double_tdc_block(x1)
        #print("x2_tdc: ", x2.size())
        x2 = self.temporal_gap(x2)
        #print("x2_gap: ", x2.size())

        if self.signal_type in ('range_doppler', 'angle_doppler'):
            # The Doppler dimension requires a specific processing
            x2 = F.pad(x2, (0, 1, 0, 0), "constant", 0)
            x2 = self.doppler_max_pool(x2)
        else:
            x2 = self.max_pool(x2)

        x2 = torch.squeeze(x2, 2) # remove temporal dimension
        #print("x2: ", x2.size())
        x3 = self.single_conv_block1_1x1(x2)
        # return input of ASPP block + latent features
        return x2, x3


class TMVA_TDC(nn.Module):
    """ 
    Temporal Multi-View with ASPP Network (TMVA-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """

    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Backbone (encoding)
        self.rd_encoding_branch = EncodingBranch('range_doppler', n_frames)
        self.ra_encoding_branch = EncodingBranch('range_angle', n_frames)
        self.ad_encoding_branch = EncodingBranch('angle_doppler', n_frames)

        # ASPP Blocks
        self.rd_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ra_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.ad_aspp_block = ASPPBlock(in_ch=128, out_ch=128)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)
        self.ad_single_conv_block1_1x1 = ConvBlock(in_ch=640, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=384, out_ch=128, k_size=1, pad=0, dil=1)

        # Pallel range-Doppler (RD) and range-angle (RA) decoding branches
        self.rd_upconv1 = nn.ConvTranspose2d(384, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(384, 128, 2, stride=2)
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)


    def forward(self, x_rd, x_ra, x_ad):
        # Backbone
        ra_features, ra_latent = self.ra_encoding_branch(x_ra)
        rd_features, rd_latent = self.rd_encoding_branch(x_rd)
        ad_features, ad_latent = self.ad_encoding_branch(x_ad)

        # ASPP blocks
        x1_rd = self.rd_aspp_block(rd_features)
        x1_ra = self.ra_aspp_block(ra_features)
        x1_ad = self.ad_aspp_block(ad_features)
        x2_rd = self.rd_single_conv_block1_1x1(x1_rd)
        x2_ra = self.ra_single_conv_block1_1x1(x1_ra)
        x2_ad = self.ad_single_conv_block1_1x1(x1_ad)

        # Latent Space
        # Features join either the RD or the RA branch
        x3 = torch.cat((rd_latent, ra_latent, ad_latent), 1)
        x3_rd = self.rd_single_conv_block2_1x1(x3)
        x3_ra = self.ra_single_conv_block2_1x1(x3)

        # Latent Space + ASPP features
        x4_rd = torch.cat((x2_rd, x3_rd, x2_ad), 1)
        x4_ra = torch.cat((x2_ra, x3_ra, x2_ad), 1)

        # Parallel decoding branches with upconvs
        x5_rd = self.rd_upconv1(x4_rd)
        x5_ra = self.ra_upconv1(x4_ra)
        x6_rd = self.rd_double_conv_block1(x5_rd)
        x6_ra = self.ra_double_conv_block1(x5_ra)

        x7_rd = self.rd_upconv2(x6_rd)
        x7_ra = self.ra_upconv2(x6_ra)
        x8_rd = self.rd_double_conv_block2(x7_rd)
        x8_ra = self.ra_double_conv_block2(x7_ra)

        # Final 1D convolutions
        x9_rd = self.rd_final(x8_rd)
        x9_ra = self.ra_final(x8_ra)

        return x9_rd, x9_ra
