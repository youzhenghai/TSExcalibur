# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import make_enc_dec
from asteroid.masknn.convolutional import TDConvNet
from models.base_models_informed import BaseEncoderMaskerDecoderInformed
from models.adapt_layers import make_adapt_layer

class Lambda(nn.Module):
    """
    https://stackoverflow.com/a/64064088
    Input: A Function
    Returns : A Module that can be used
        inside nn.Sequential
    """
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, x): return self.func(x, **self.kwargs)

class SpEx_Plus(BaseEncoderMaskerDecoderInformed):
    """TimeDomain SpeakerBeam target speech extraction model.
    Adapted from Asteroid class ConvTasnet 
    https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/conv_tasnet.py

    Args:
        i_adapt_layer (int): Index of adaptation layer.
        adapt_layer_type (str): Type of adaptation layer, see adapt_layers.py for options.
        adapt_enroll_dim (int): Dimensionality of the speaker embedding.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        causal (bool, optional) : Whether or not the convolutions are causal.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.
    """

    # 1. 对齐参数
    def __init__(
        self,
        out_chan=None,
        n_blocks=8,
        n_repeats=4,
        causal=False,       
        norm_type="gLN",
        in_chan=None,
        fb_name="free",
        L1=20,
        L2=80,
        L3=160,
        n_filters=256,
        encoder_activation=None,
        sample_rate=8000,
        num_spks=101,
        embeddings_size=256,
        fusion_type='cat',
        O=256,  # bn_chan + skip_chan
        P=512,  # hid_chan
        **fb_kwargs,
    ):
        
        encoder,decoder =  CombinedEncoder(

        )
        #instancenorm = nn.InstanceNorm1d(n_filters)

        # n_feats = encoder_1d_short.n_feats_out
        # if in_chan is not None:
        #     assert in_chan == n_feats, (
        #         "Number of filterbank output channels"
        #         " and number of input channels should "
        #         "be the same. Received "
        #         f"{n_feats} and {in_chan}"
        #     )
        masker = SpExInformed(
            n_feats,
            fusion_type,
            embeddings_size,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            norm_type=norm_type,
            causal=causal,
            #mask_act='linear',  # ????????
            #conv_kernel_size=conv_kernel_size,
            out_chan=out_chan,
            bn_chan=O,
            hid_chan=P,
            O=O,
            P=P,
        )

        super().__init__(encoder, masker, decoder, 
                         encoder_activation=encoder_activation)



class CombinedEncoder(nn.Module):
    def __init__(
        self,
        fb_name,
        L1,
        L2,
        L3,
        n_filters,
        sample_rate,
        O=O,
        is_innorm=False,
    ):
        super(CombinedEncoder, self).__init__()

    # 2. make_enc_dec 和 Conv1D对应吗
        # encoder_1d_short, decoder_1d_short = make_enc_dec(
        #     fb_name,
        #     kernel_size=L1,
        #     n_filters=n_filters,
        #     stride=L1 // 2,
        #     sample_rate=sample_rate,
        #     **fb_kwargs,
        # )

        self.encoder_1d_short = Conv1D(1, n_filters, L1, stride=L1 // 2, padding=0)

        # encoder_1d_middle, decoder_1d_middle = make_enc_dec(
        #     fb_name,
        #     kernel_size=L2,
        #     n_filters=n_filters,
        #     stride=L1 // 2,
        #     sample_rate=sample_rate,
        #     **fb_kwargs,
        # )
        self.encoder_1d_middle = Conv1D(1, n_filters, L2, stride=L1 // 2, padding=0)
        # encoder_1d_long, decoder_1d_long = make_enc_dec(
        #     fb_name,
        #     kernel_size=L3,
        #     n_filters=n_filters,
        #     stride=L1 // 2,
        #     sample_rate=sample_rate,
        #     **fb_kwargs,
        # )
        self.encoder_1d_long = Conv1D(1, n_filters, L3, stride=L1 // 2, padding=0)

        if is_innorm:
            self.instancenorm = nn.InstanceNorm1d(n_filters)


    def forward(self, x):

        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)

        w1 = F.relu(self.encoder_1d_short(x))
        T = w1.shape[-1]
        xlen1 = x.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        w2 = F.relu(self.encoder_1d_middle(F.pad(x, (0, xlen2 - xlen1), "constant", 0)))
        w3 = F.relu(self.encoder_1d_long(F.pad(x, (0, xlen3 - xlen1), "constant", 0)))

        if self.is_norm:
            w1 = self.instancenorm(w1)
            w2 = self.instancenorm(w2)
            w3 = self.instancenorm(w3)            

        # y = self.ln(torch.cat([w1, w2, w3], 1))
        # y = self.proj(y)

        return w1,w2,w3,T





class CombinedDecoder(nn.Module):
    def __init__(
        self,
        fb_name,
        L1,
        L2,
        L3,
        n_filters,
        sample_rate,
        O,
        is_innorm=False,
    ):
        super(CombinedEncoder, self).__init__()
        self.L1=L1
        self.L2=L2
        self.L3=L3
        self.decoder_1d_short = ConvTrans1D(n_filters, 1, kernel_size=L1, stride=L1 // 2, bias=True)
        self.decoder_1d_middle = ConvTrans1D(n_filters, 1, kernel_size=L2, stride=L1 // 2, bias=True)
        self.decoder_1d_long = ConvTrans1D(n_filters, 1, kernel_size=L3, stride=L1 // 2, bias=True)
 
    def forward(self, S1,S2,S3,L):
        T = S1.shape[-1]
        xlen1 = L # L = x.shape[-1]
        xlen2 = (T - 1) * (self.L1 // 2) + self.L2
        xlen3 = (T - 1) * (self.L1 // 2) + self.L3
        
        return self.decoder_1d_short(S1), self.decoder_1d_middle(S2)[:, :xlen1], self.decoder_1d_long(S3)[:, :xlen1]



class TDConvNetInformed(TDConvNet):
    """
    Adapted from Asteroid class TDConvNet 
    https://github.com/asteroid-team/asteroid/blob/master/asteroid/masknn/convolutional.py
    """
    def __init__(
        self,
        in_chan,
        i_adapt_layer,
        adapt_layer_type,
        adapt_enroll_dim,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
        **adapt_layer_kwargs
    ):
        super(TDConvNetInformed, self).__init__(
                in_chan, 1, out_chan, n_blocks, n_repeats, 
                bn_chan, hid_chan, skip_chan, conv_kernel_size,
                norm_type, mask_act, causal)
        self.i_adapt_layer = i_adapt_layer
        self.adapt_enroll_dim = adapt_enroll_dim
        self.adapt_layer_type = adapt_layer_type
        self.adapt_layer = make_adapt_layer(adapt_layer_type, 
                                            indim=bn_chan,
                                            enrolldim=adapt_enroll_dim,
                                            ninputs=2 if self.skip_chan else 1,
                                            **adapt_layer_kwargs)

    def forward(self, mixture_w, enroll_emb):
        r"""Forward with auxiliary enrollment information
        
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
            enroll_emb (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
                                                or $(batch, nfilters)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, 1, nfilters, nframes)$
        """
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for i, layer in enumerate(self.TCN):
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                if i == self.i_adapt_layer:
                    residual, skip = self.adapt_layer((residual, skip), 
                                            torch.chunk(enroll_emb,2,dim=1))
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
                if i == self.i_adapt_layer:
                    residual = self.adapt_layer(residual, enroll_emb)
            output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, 1, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'i_adapt_layer': self.i_adapt_layer,
            'adapt_layer_type': self.adapt_layer_type,
            'adapt_enroll_dim': self.adapt_enroll_dim
            })
        return config


class Extractor(nn.Module):
    def __init__(self,
                 L1=20,
                 L2=80,
                 L3=160,
                 N=256,
                 B=8,
                 O=256,
                 P=512,
                 Q=3,
                 num_spks=101,
                 spk_embed_dim=256,
                 causal=False,
                 fusion_type='cat',
                 norm_type='gLN',
                 ):
        super(Extractor, self).__init__()
        # n x N x T => n x O x T
        self.ln = ChannelwiseLayerNorm(3*N)
        self.proj = Conv1D(3*N, O, 1)
        self.conv_block_1 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal, dilation=1,fusion_type=fusion_type,norm_type=norm_type)
        self.conv_block_1_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal,norm_type=norm_type)
        self.conv_block_2 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal, dilation=1,fusion_type=fusion_type,norm_type=norm_type)
        self.conv_block_2_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal,norm_type=norm_type)
        self.conv_block_3 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal, dilation=1,fusion_type=fusion_type,norm_type=norm_type)
        self.conv_block_3_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal,norm_type=norm_type)
        self.conv_block_4 = TCNBlock_Spk(spk_embed_dim=spk_embed_dim, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal, dilation=1,fusion_type=fusion_type,norm_type=norm_type)
        self.conv_block_4_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal,norm_type=norm_type)
        # n x O x T => n x N x T
        self.mask1 = Conv1D(O, N, 1)
        self.mask2 = Conv1D(O, N, 1)
        self.mask3 = Conv1D(O, N, 1)

    def _build_stacks(self, num_blocks, **block_kwargs):
        """
        Stack B numbers of TCN block, the first TCN block takes the speaker embedding
        """
        blocks = [
            TCNBlock(**block_kwargs, dilation=(2**b))
            for b in range(1,num_blocks)
        ]
        return nn.Sequential(*blocks)

    def forward(self, w1, w2, w3, aux):

        y = self.ln(th.cat([w1, w2, w3], 1))
        # n x O x T
        y = self.proj(y)
        y = self.conv_block_1(y, aux)
        y = self.conv_block_1_other(y)
        y = self.conv_block_2(y, aux)
        y = self.conv_block_2_other(y)
        y = self.conv_block_3(y, aux)
        y = self.conv_block_3_other(y)
        y = self.conv_block_4(y, aux)
        y = self.conv_block_4_other(y)

        # n x N x T
        m1 = F.relu(self.mask1(y))
        m2 = F.relu(self.mask2(y))
        m3 = F.relu(self.mask3(y))

        return m1, m2, m3
    

class Speaker_Model(nn.Module):
    def __init__(self,
                 L1=20,
                 L2=80,
                 L3=160,
                 N=256,
                 O=256,
                 P=512,
                 spk_embed_dim=256,
                ):
        super(Speaker_Model, self).__init__()
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.spk_encoder = nn.Sequential(
            ChannelwiseLayerNorm(3*N),
            Conv1D(3*N, O, 1),
            ResBlock(O, O),
            ResBlock(O, P),
            ResBlock(P, P),
            Conv1D(P, spk_embed_dim, 1),
        )
    def forward(self, aux, aux_len):
        aux = self.spk_encoder(aux)
        aux_T = (aux_len - self.L1) // (self.L1 // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = th.sum(aux, -1)/aux_T.view(-1,1).float()   
        return aux