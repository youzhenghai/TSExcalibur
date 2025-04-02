from asteroid_filterbanks import make_enc_dec
# from ..masknn import DPRNN
from .tse_models import BaseEncoderMaskerDecoderInformed
import torch
from torch import nn
from torch.nn.functional import fold, unfold
from .norms import GlobLN

class ResBlock(nn.Module):
    ''' Resnet block for speaker encoder to obtain speaker embedding.

    Ref to:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://github.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    '''
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpool = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return self.maxpool(y)


class SingleRNN(nn.Module):
    ''' Single RNN layer.

    Args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int.
        hidden_size: int.
        dropout: float.
        bidirectional: bool.
    '''
    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.rnn = getattr(nn, rnn_type)(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=1,
                                         dropout=dropout,
                                         batch_first=True,
                                         bidirectional=bidirectional)

    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, input):
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = input
        rnn_output, _ = self.rnn(output)
        return rnn_output

class DPRNNBlock(nn.Module):
    ''' Dual-Path RNN Block.

    Args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        feature_size: int.
        hidden_size: int.
        norm_type: string, 'gLN' or 'ln'.
        dropout: float.
        bidirectional: bool.
    '''
    def __init__(self, rnn_type, feature_size, hidden_size,
                 norm_type='gLN', dropout=0, bidirectional=True):
        super().__init__()

        self.intra_rnn = SingleRNN(
            rnn_type,
            feature_size,
            hidden_size,
            dropout=dropout,
            bidirectional=True, # always bi-directional
        )
        self.intra_linear = nn.Linear(self.intra_rnn.output_size(), feature_size)

        self.inter_rnn = SingleRNN(
            rnn_type,
            feature_size,
            hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.inter_linear = nn.Linear(self.inter_rnn.output_size(), feature_size)

        if norm_type == 'gLN':
            self.intra_norm = GlobLN(feature_size)
            self.inter_norm = GlobLN(feature_size)
        if norm_type == 'ln':
            self.intra_norm = nn.GroupNorm(1, feature_size)
            self.inter_norm = nn.GroupNorm(1, feature_size)

    def forward(self, input):
         # input: [B, N, K, S]
        B, N, K, S = input.size()
        output = input
        # Intra-chunk processing
        input = input.transpose(1, -1).reshape(B * S, K, N)
        # -> [BS, K, N]
        input = self.intra_rnn(input)
        # -> # [BS, K, N]
        input = self.intra_linear(input)
        input = input.reshape(B, S, K, N).transpose(1, -1)
        input = self.intra_norm(input)
        # residual connection
        output = output + input
        # Inter-chunk processing
        input = output.transpose(1, 2).transpose(2, -1).reshape(B * K, S, N)
        input = self.inter_rnn(input)
        input = self.inter_linear(input)
        input = input.reshape(B, K, S, N).transpose(1, -1).transpose(2, -1).contiguous()
        input = self.inter_norm(input)
        return output + input

class DPRNN(nn.Module):
    ''' Dual-Path RNN.

    Args:
        input_size: int.
        feature_size: int.
        hidden_size: int.
        chunk_length: int.
        hop_length: int.
        n_repeats: int.
        bidirectional: bool.
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        norm_type: string, 'gLN' or 'ln'.
        activation_type: string, 'sigmoid' or 'relu'.
        dropout: float.
    '''
    def __init__(self, input_size, feature_size=128, hidden_size=128,
                 chunk_length=200, hop_length=None, n_repeats=6,
                 bidirectional=True, rnn_type='LSTM', norm_type='gLN',
                 activation_type='sigmoid', dropout=0):
        super().__init__()
        
        self.input_size = input_size
        self.feature_size = feature_size
        # length of chunk in segmentation
        self.chunk_length = chunk_length
        # length of hop in segmentation
        self.hop_length = hop_length if hop_length is not None else chunk_length // 2

        # bottleneck
        if norm_type == 'gLN':
            linear_norm = GlobLN(input_size)
        if norm_type == 'ln':
            linear_norm = nn.GroupNorm(1, input_size)
        start_conv1d = nn.Conv1d(input_size, feature_size, 1)
        self.bottleneck = nn.Sequential(linear_norm, start_conv1d)

        # stack dprnn blocks
        dprnn_blocks = []
        for _ in range(n_repeats):
            dprnn_blocks += [
                DPRNNBlock(
                    rnn_type=rnn_type,
                    feature_size=feature_size,
                    hidden_size=hidden_size,
                    norm_type=norm_type,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            ]
        self.dprnn_blocks = nn.Sequential(*dprnn_blocks)

        # masks
        self.prelu = nn.PReLU()
        self.conv2d = nn.Conv2d(feature_size, feature_size, kernel_size=1)

        self.out = nn.Sequential(nn.Conv1d(feature_size, feature_size, 1), nn.Tanh())
        self.gate = nn.Sequential(nn.Conv1d(feature_size, feature_size, 1), nn.Sigmoid())

        self.end_conv1x1 = nn.Conv1d(feature_size, input_size, 1, bias=False)
        if activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        if activation_type == 'relu':
            self.activation = nn.ReLU()

    def forward(self, input):
        # input: [B, N(input_size), L]
        B, _, L = input.size()
        output = self.bottleneck(input)
        # -> [B, N(feature_size), L]
        output, n_chunks = self._segmentation(output)
        # ->  # [B, N, K, S]
        output = self.dprnn_blocks(output)
        output = self.prelu(output) 
        output = self.conv2d(output)
        # ->  # [B,N(feature_size), K, S]
        output = output.reshape(B, self.feature_size, self.chunk_length, n_chunks)
        # -> [B, N(feature_size), K, S]
        output = self._overlap_add(output, L)
        # -> [B, N(feature_size), L]
        output = self.out(output) * self.gate(output)
        output = self.end_conv1x1(output)
        # -> [B, N(input_size), L]
        output = self.activation(output)
        output = output.reshape(B, 1, self.input_size, L)
        # -> [B, 1, N(input_size), L]
        return output

    def _segmentation(self, input):
        # input: B, N(feature_size), L]
        B, _, _ = input.size()
        output = unfold(
            input.unsqueeze(-1),
            kernel_size=(self.chunk_length, 1),
            padding=(self.chunk_length, 0),
            stride=(self.hop_length, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(B, self.feature_size, self.chunk_length, n_chunks)
        # -> [B, N(feature_size), K, S]
        return output, n_chunks

    def _overlap_add(self, input, L):
        # input: [2 * B, N(feature_size), K, S]
        batchx2, _, _, n_chunks = input.size()
        output = input
        to_unfold = self.feature_size * self.chunk_length
        output = fold(
            output.reshape(batchx2, to_unfold, n_chunks),
            (L, 1),
            kernel_size=(self.chunk_length, 1),
            padding=(self.chunk_length, 0),
            stride=(self.hop_length, 1),
        )
        output = output.reshape(batchx2, self.feature_size, -1)
        # -> [B, N(feature_size), L]
        return output

class DPRNNSpe(DPRNN):
    ''' Dual-Path RNN. (Spe)

    Args:
        input_size: int.
        feature_size: int.
        hidden_size: int.
        chunk_length: int.
        hop_length: int.
        n_repeats: int.
        bidirectional: bool.
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        norm_type: string, 'gLN' or 'ln'.
        activation_type: string, 'sigmoid' or 'relu'.
        dropout: float.
        O: int.
        P: int.
        embeddings_size: int.
        num_spks: int.
        kernel_size: int.
        fusion_type: string, 'cat', 'add', 'mul' or 'film'.
    '''
    def __init__(self, input_size, feature_size=128, hidden_size=128,
                 chunk_length=200, hop_length=None, n_repeats=6,
                 bidirectional=True, rnn_type='LSTM', norm_type='gLN',
                 activation_type='sigmoid', dropout=0, O=128, P=256,
                 embeddings_size=128, num_spks=251, kernel_size=2,
                 fusion_type='cat'):
        super().__init__(
            input_size,
            feature_size,
            hidden_size,
            chunk_length,
            hop_length,
            n_repeats,
            bidirectional,
            rnn_type,
            norm_type,
            activation_type,
            dropout=dropout,
        )
        self.input_size = input_size
        self.feature_size = feature_size
        self.chunk_length = chunk_length
        self.hidden_size = hidden_size
        self.hop_length = hop_length
        self.n_repeats = n_repeats
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.norm_type = norm_type
        self.activation_type = activation_type
        self.dropout = dropout
        self.O = O
        self.P = P
        self.embeddings_size = embeddings_size
        self.num_spks = num_spks
        self.kernel_size = kernel_size
        self.fusion_type = fusion_type


        # fusion
        if fusion_type == 'cat':
            start_conv1d = nn.Conv1d(input_size + embeddings_size, feature_size, 1)
        if fusion_type in ('add', 'mul'):
            self.fusion_linear = nn.Linear(embeddings_size, input_size)
            start_conv1d = nn.Conv1d(input_size, feature_size, 1)
        if fusion_type == 'film':
            self.fusion_linear_1 = nn.Linear(embeddings_size, input_size)
            self.fusion_linear_2 = nn.Linear(embeddings_size, input_size)
            start_conv1d = nn.Conv1d(input_size, feature_size, 1)
        if fusion_type == 'att':
            self.fusion_linear = nn.Linear(embeddings_size, input_size)
            self.average = nn.Conv1d(input_size, input_size, kernel_size, kernel_size, groups=input_size)
            self.average.weight = nn.Parameter(torch.ones(input_size, 1, kernel_size) / kernel_size)
            self.average.bias = nn.Parameter(torch.zeros(input_size))
            for p in self.average.parameters():
                p.requires_grad = False
            start_conv1d = nn.Conv1d(input_size, feature_size, 1)

        # bottleneck
        if norm_type == 'gLN':
            linear_norm = GlobLN(input_size)
        else:
            linear_norm = nn.GroupNorm(1, input_size)
        self.bottleneck = nn.Sequential(linear_norm, start_conv1d)

        # target
        self.spk_encoder = nn.Sequential(
            nn.GroupNorm(1, input_size),
            nn.Conv1d(input_size, O, 1),
            ResBlock(O, O),
            ResBlock(O, P),
            ResBlock(P, P),
            nn.Conv1d(P, embeddings_size, 1),
        )
        self.pred_linear = nn.Linear(embeddings_size, num_spks)

    def forward(self, input, aux, aux_len):
        # input: [B, N(input_size), L]
        # aux: [B, N(input_size), L]

        B, _, L = input.size()

        # auxiliary
        aux = self._auxiliary(aux, aux_len)
        # -> [B, N(embeddings_size)]

        # norm
        output = self.bottleneck[0](input)
        # -> [B, N(input_size), L]

        # fusion
        output = self._fusion(aux, output, L)

        # 1x1 cnn
        output = self.bottleneck[1](output)
        # -> [B, N(feature_size), L]

        # dprnn blocks
        output = self._dprnn_process(output, B, L)
        # -> [B, 1, N(input_size), L]

        # auxiliary linear
        aux = self.pred_linear(aux)
        # -> [B, num_spks]

        return output, aux

    def _auxiliary(self, aux, aux_len):
        aux = self.spk_encoder(aux)
        # -> [B, embeddings_size, L // 3] * 3 resnet
        aux_T = (aux_len - self.kernel_size) // (self.kernel_size // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = torch.sum(aux, -1) / aux_T.view(-1, 1).float()
        # -> [B, embeddings_size]
        return aux

    def _fusion(self, aux, output, L):
        if self.fusion_type == 'cat':
            output = self._concatenation(aux, output, L)
            # -> [B, N(input_size + embeddings_size), L]
        if self.fusion_type == 'add':
            output = self._addition(aux, output, L, self.fusion_linear)
            # -> [B, N(input_size), L]
        if self.fusion_type == 'mul':
            output = self._multiplication(aux, output, L, self.fusion_linear)
            # -> [B, N(input_size), L]
        if self.fusion_type == 'film':
            output = self._film(aux, output, L)
            # -> [B, N(input_size), L]
        if self.fusion_type == 'att':
            print("output", output.shape)
            output_avg = self.average(output)
            print("output_avg", output_avg.shape)
            att_out = self._attention(aux, output_avg, self.fusion_linear)
            print("att_out", att_out.shape)
            upsampling = nn.Upsample(size=L, mode='nearest')
            att_out = upsampling(att_out)
            print("upsampling", att_out.shape)
            output = output * att_out
        return output

    def _concatenation(self, aux, output, L):
        aux_concat = torch.unsqueeze(aux, -1)
        aux_concat  = aux_concat.repeat(1, 1, L)
         # -> [B, N(embeddings_size), L]
        output = torch.cat([output, aux_concat], 1)
        # -> [B, N(input_size + embeddings_size), L]
        return output

    def _addition(self, aux, output, L, fusion_linear):
        aux_add = fusion_linear(aux)
        # -> [B, N(input_size)]
        aux_add = torch.unsqueeze(aux_add, -1)
        aux_add = aux_add.repeat(1, 1, L)
        # -> [B, N(input_size), L]
        output = output + aux_add
        # -> [B, N(input_size, L]
        return output

    def _multiplication(self, aux, output, L, fusion_linear):
        aux_mul = fusion_linear(aux)
        # -> [B, N(input_size)]
        aux_mul = torch.unsqueeze(aux_mul, -1)
        aux_mul = aux_mul.repeat(1, 1, L)
        # -> [B, N(input_size), L]
        output = output * aux_mul
        # -> [B, N(input_size, L]
        return output
    
    def _attention(self, aux, output, fusion_linear):
        L = output.shape[-1]
        aux_att = fusion_linear(aux) 
        print("aux_att", aux_att.shape)
        aux_att = torch.unsqueeze(aux_att, -1)
        aux_att = aux_att.repeat(1, 1, L)
        att = torch.sum(output * aux_att, 1, keepdim=True) 
        att = F.softmax(att, -1)
        att = att * aux_att
        return att + aux_att 

    def _film(self, aux, output, L):
        output = self._multiplication(aux, output, L, self.fusion_linear_1)
        # -> [B, N(input_size, L]
        output = self._addition(aux, output, L, self.fusion_linear_2)
        # -> [B, N(input_size, L]
        return output

    def _dprnn_process(self, output, B, L):
        output, n_chunks = self._segmentation(output)
        # ->  # [B, N, K, S]
        output = self.dprnn_blocks(output)
        output = self.prelu(output)
        output = self.conv2d(output)
        # ->  # [B, 1 * N(feature_size), K, S]
        output = output.reshape(B, self.feature_size, self.chunk_length, n_chunks)
        # -> [1 * B, N(feature_size), K, S]
        output = self._overlap_add(output, L)
        # -> [1 * B, N(feature_size), L]
        output = self.out(output) * self.gate(output)
        output = self.end_conv1x1(output)
        # -> [1 * B, N(input_size), L]
        output = self.activation(output)
        output = output.reshape(B, 1, self.input_size, L)
        # -> [B, 1, N(input_size), L]
        return output


    def get_config(self):
        config = {
            "in_chan": self.input_size,
            "out_chan": self.input_size,
            "bn_chan": self.feature_size,
            "hid_size": self.hidden_size,
            "chunk_size": self.chunk_length,
            "hop_size": self.hop_length,
            "n_repeats": self.n_repeats,
            "n_src": 1,
            "norm_type": self.norm_type,
            "mask_act": self.activation_type,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "num_layers": 1,
            "dropout": self.dropout,
            "use_mulcat": False,
            'O': self.O,
            'P': self.P,
            'embeddings_size': self.embeddings_size,
            'num_spks': self.num_spks,
            'fusion_type': self.fusion_type,
        }
        print('num_spks',self.num_spks)
        return config


class DPRNNSpeTasNet(BaseEncoderMaskerDecoderInformed):
    """DPRNN separation model, as described in [1].

    Args:
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        bn_chan (int): Number of channels after the bottleneck.
            Defaults to 128.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use. To choose from

            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
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

    References
        - [1] "Dual-path RNN: efficient long sequence modeling for
          time-domain single-channel speech separation", Yi Luo, Zhuo Chen
          and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """

    def __init__(
        self,
        out_chan=None,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="sigmoid",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        in_chan=None,
        fb_name="free",
        kernel_size=16,
        n_filters=64,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        use_mulcat=False,
        num_spks=251,
        embeddings_size=128,
        fusion_type='cat',
        O=256,
        P=512,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        # Update in_chan
        masker = DPRNNSpe(
            input_size=out_chan,
            feature_size=bn_chan,
            hidden_size=hid_size,
            chunk_length=chunk_size,
            hop_length=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            activation_type=mask_act,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            dropout=dropout,
            num_spks= num_spks,
            O=O,
            P=P,
            embeddings_size=embeddings_size,
            fusion_type=fusion_type,
            kernel_size=kernel_size,
        )

        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
