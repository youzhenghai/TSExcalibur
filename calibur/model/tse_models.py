import torch
from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape
from asteroid.models.base_models import _shape_reconstructed, _unsqueeze_to_3d
from asteroid.models.base_models import BaseEncoderMaskerDecoder

class BaseEncoderMaskerDecoderInformed(BaseEncoderMaskerDecoder):
    """Base class for informed encoder-masker-decoder extraction models.
    Adapted from Asteroid calss BaseEncoderMaskerDecoder
    https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/base_models.py

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masked network.
        decoder (Decoder): Decoder instance.
        auxiliary (nn.Module): auxiliary network processing enrollment.
        encoder_activation (optional[str], optional): activation to apply after encoder.
            see ``asteroid.masknn.activations`` for valid values.
    """

    def forward(self, wav, enrollment, aux_len):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            enrollment (torch.Tensor): enrollment information tensor. 1D, 2D or 3D tensor.

        Returns:
            torch.Tensor, of shape (batch, 1, time) or (1, time)
        """

        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)

        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)
        enrollment = _unsqueeze_to_3d(enrollment)

        # print("wav.shape: ", wav.shape)
        # print("enrollment.shape: ", enrollment.shape)
        # print("aux_len: ", aux_len.shape)
        # Real forward
        tf_rep = self.forward_encoder(wav)
        tf_aux = self.forward_encoder(enrollment)
        est_masks, aux = self.masker(tf_rep, tf_aux, aux_len)

        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape), aux


    # def forward_masker(self, tf_rep: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
    #     """Estimates masks from time-frequency representation.

    #     Args:
    #         tf_rep (torch.Tensor): Time-frequency representation in (batch,
    #             feat, seq).
    #         enroll (torch.Tensor): Time-frequency representation in (batch,
    #             feat, seq).

    #     Returns:
    #         torch.Tensor: Estimated masks
    #     """
    #     return self.masker(tf_rep, enroll)