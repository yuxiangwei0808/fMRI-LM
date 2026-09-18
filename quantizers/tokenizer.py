import math
from typing import Mapping, Text, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import inspect
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import OmegaConf
from functools import partial
from accelerate.utils.operations import gather

from brain_encoder.patch_embed import PatchEmbed
from brain_encoder.vision_transformer import Block


class VectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        use_l2_norm: bool = False,
        clustering_vq: bool = False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost
        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm
        self.clustering_vq = clustering_vq
        if clustering_vq:
            self.decay = 0.99
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
            z_flattened = rearrange(z, 'b h w c -> (b h w) c')
            unnormed_z_flattened = z_flattened

            if self.use_l2_norm:
                z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
                embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
            else:
                embedding = self.embedding.weight

            d = (
                torch.sum(z_flattened ** 2, dim=1, keepdim=True)
                + torch.sum(embedding ** 2, dim=1)
                - 2 * torch.einsum('bd,dn->bn', z_flattened, embedding.T)
            )
            min_encoding_indices = torch.argmin(d, dim=1)
            z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

            if self.use_l2_norm:
                z = torch.nn.functional.normalize(z, dim=-1)

            commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) ** 2)
            codebook_loss = torch.mean((z_quantized - z.detach()) ** 2)

            if self.clustering_vq and self.training:
                with torch.no_grad():
                    encoding_indices = gather(min_encoding_indices)
                    if len(min_encoding_indices.shape) != 1:
                        raise ValueError(f"min_encoding_indices in a wrong shape, {min_encoding_indices.shape}")
                    encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
                    encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                    avg_probs = torch.mean(encodings, dim=0)
                    self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)

                    all_d = gather(d)
                    all_unnormed_z_flattened = gather(unnormed_z_flattened).detach()
                    if all_d.shape[0] != all_unnormed_z_flattened.shape[0]:
                        raise ValueError(
                            "all_d and all_unnormed_z_flattened have different length"
                            f"{all_d.shape}, {all_unnormed_z_flattened.shape}"
                        )
                    indices = torch.argmin(all_d, dim=0)
                    random_feat = all_unnormed_z_flattened[indices]
                    decay = torch.exp(
                        -(self.embed_prob * self.codebook_size * 10) / (1 - self.decay) - 1e-3
                    ).unsqueeze(1).repeat(1, self.token_size)
                    self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

            loss = commitment_loss + codebook_loss
            z_quantized = z + (z_quantized - z).detach()
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

            result_dict = dict(
                quantizer_loss=loss,
                commitment_loss=commitment_loss,
                codebook_loss=codebook_loss,
                min_encoding_indices=min_encoding_indices.view(
                    z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3]
                ),
            )
            return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        return self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)

    def mode(self):
        return self.mean

    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        return 0.5 * torch.sum(
            torch.pow(self.mean.float(), 2) + self.var.float() - 1.0 - self.logvar.float(),
            dim=[1, 2],
        )


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

def _exists(v):
    return v is not None

def _default(*args):
    for arg in args:
        if _exists(arg):
            return arg
    return None

def _as_bool(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "true", "yes", "y", "on"):
            return True
        if value in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"Cannot parse boolean value: {value}")
    return bool(value)

def _round_ste(z):
    """Round with straight-through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class FSQ(nn.Module):
    """Finite scalar quantizer with the same image-shaped interface as VQ."""

    def __init__(
        self,
        levels,
        dim=None,
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        scale=None,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + list(levels[:-1]), dtype=torch.int32), dim=0)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale
        self.codebook_dim = len(levels)
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = self.codebook_dim * num_codebooks

        keep_num_codebooks_dim = _default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = _default(dim, self.effective_codebook_dim)
        has_projections = self.dim != self.effective_codebook_dim
        self.project_in = nn.Linear(self.dim, self.effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(self.effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z, eps=1e-3):
        """Bound z to the valid scalar ranges before rounding."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        quantized = _round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat):
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.long)

    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z):
        is_img_or_video = z.ndim >= 4

        if is_img_or_video:
            batch_size, channels, *spatial_shape = z.shape
            z = z.reshape(batch_size, channels, -1).permute(0, 2, 1).contiguous()

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found {z.shape[-1]}"

        z = self.project_in(z)
        z = rearrange(z, 'b n (c d) -> b n c d', c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')
        out = self.project_out(codes)

        if is_img_or_video:
            out = out.permute(0, 2, 1).reshape(batch_size, self.dim, *spatial_shape).contiguous()
            if self.keep_num_codebooks_dim:
                indices = indices.reshape(batch_size, *spatial_shape, self.num_codebooks)
            else:
                indices = indices.reshape(batch_size, *spatial_shape)
        elif not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return out, indices

# Adapted from NPQ-ViT's BinarySphericalQuantizer to match this Tokenizer's
# BCHW tensor layout and result_dict interface.
class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        bits = (zq > 0).to(torch.long)
        zi = (bits * basis).sum(dim=-1)
        cnt = torch.zeros(2 ** K, device=zq.device, dtype=zq.dtype)
        cnt.scatter_add_(0, zi.flatten(), torch.ones_like(zi.flatten(), dtype=zq.dtype))
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None


def _codebook_entropy(zq, basis, K, eps=1e-4):
    if K > 24:
        raise ValueError(
            "Hard BSQ entropy allocates 2**token_size bins. "
            "Use soft_entropy=True or token_size <= 24."
        )
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


class BinarySphericalQuantizer(nn.Module):
    """Binary spherical quantizer used by NPQ-style tokenizers.

    The quantized latent is a sign vector in {-1, 1}. The full codebook index
    packs the token_size bits into a single int64 label, matching the existing
    get_codebook_indices contract used by the fMRI language-model objective.
    """

    def __init__(
        self,
        embed_dim,
        beta=0.0,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        input_format="bchw",
        soft_entropy=True,
        group_size=1,
        index_mode="full",
        persample_entropy_compute="group",
        cb_entropy_compute="group",
        l2_norm=False,
        inv_temperature=1.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.beta = float(beta)
        self.gamma0 = float(gamma0)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.input_format = input_format
        self.soft_entropy = _as_bool(soft_entropy)
        self.group_size = int(group_size)
        self.index_mode = str(index_mode).lower()
        self.l2_norm = _as_bool(l2_norm)
        self.inv_temperature = float(inv_temperature)

        if self.embed_dim <= 0:
            raise ValueError("BSQ embed_dim/token_size must be positive.")
        if self.index_mode not in ("full", "grouped"):
            raise ValueError("BSQ index_mode must be 'full' or 'grouped'.")
        if self.index_mode == "full" and self.embed_dim > 63:
            raise ValueError(
                "BSQ full codebook indices are packed into torch.int64, so "
                "model.token_size must be <= 63. Set model.bsq_index_mode='grouped' "
                "for tokenizer-only training with larger token_size."
            )
        if self.group_size <= 0 or self.embed_dim % self.group_size != 0:
            raise ValueError("BSQ group_size must be positive and divide token_size.")
        if self.group_size > 20:
            raise ValueError("BSQ group_size builds a 2**group_size entropy table; use group_size <= 20.")
        if input_format not in ("bchw", "blc"):
            raise ValueError("BSQ input_format must be 'bchw' or 'blc'.")
        if persample_entropy_compute not in ("group", "analytical"):
            raise ValueError("persample_entropy_compute must be 'group' or 'analytical'.")
        if cb_entropy_compute not in ("group", "nce"):
            raise ValueError("cb_entropy_compute must be 'group' or 'nce'.")

        self.num_groups = self.embed_dim // self.group_size
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute

        if self.embed_dim <= 63:
            basis = 2 ** torch.arange(self.embed_dim - 1, -1, -1, dtype=torch.long)
        else:
            basis = torch.empty(0, dtype=torch.long)
        group_basis = 2 ** torch.arange(self.group_size - 1, -1, -1, dtype=torch.long)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("group_basis", group_basis, persistent=False)

        self.codebook_size = 2 ** (self.embed_dim if self.index_mode == "full" else self.group_size)
        self.num_dimensions = 2 ** self.embed_dim
        self.bits_per_index = self.embed_dim if self.index_mode == "full" else self.group_size

        group_codes = torch.arange(2 ** self.group_size, dtype=torch.long).unsqueeze(-1)
        group_codebook = torch.remainder(torch.floor_divide(group_codes, group_basis), 2)
        group_codebook = group_codebook.float() * 2 - 1
        self.register_buffer("group_codebook", group_codebook, persistent=False)

    def quantize(self, z):
        if z.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected {self.embed_dim} dimensions, got {z.shape[-1]}")
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        return z + (zhat - z).detach()

    def forward(self, z):
        if self.input_format == "bchw":
            z = rearrange(z, "b c h w -> b h w c")

        zq = self.quantize(z)
        group_indices = self.codes_to_group_indexes(zq.detach())
        if self.index_mode == "full":
            full_indices = self.codes_to_indexes(zq.detach())
            min_encoding_indices = full_indices
        else:
            full_indices = None
            min_encoding_indices = group_indices
        used_codes = torch.unique(min_encoding_indices, return_counts=False) if not self.training else None

        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            avg_prob = None
            zb_by_sample = ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = _codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

        q_scale = self.embed_dim ** -0.5 if self.l2_norm else 1.0
        zq = zq * q_scale
        commitment_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))
        entropy_loss = self.zeta * entropy_penalty / self.inv_temperature
        quantizer_loss = commitment_loss + entropy_loss

        if self.input_format == "bchw":
            zq = rearrange(zq, "b h w c -> b c h w").contiguous()

        result_dict = {
            "quantizer_loss": quantizer_loss,
            "commitment_loss": commitment_loss,
            "entropy_loss": entropy_loss,
            "H": cb_entropy,
            "used_codes": used_codes,
            "indices": min_encoding_indices,
            "full_indices": full_indices,
            "min_encoding_indices": min_encoding_indices,
            "group_indices": group_indices,
            "avg_prob": avg_prob,
        }
        return zq, result_dict

    def soft_entropy_loss(self, z):
        group_codebook = self.group_codebook.to(device=z.device, dtype=z.dtype)
        if self.l2_norm:
            group_codebook = group_codebook / (self.embed_dim ** 0.5)

        divided_z = rearrange(z, "... (g c) -> ... g c", c=self.group_size)
        distance = -2 * torch.einsum("... g c, d c -> ... g d", divided_z, group_codebook)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)

        if self.persample_entropy_compute == "analytical":
            scale = self.embed_dim ** 0.5 if self.l2_norm else 1.0
            p = torch.sigmoid(-4 * z / scale * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        avg_prob = reduce(prob, "... g d -> g d", "mean")
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample):
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = -probs_per_dim * torch.log(probs_per_dim + 1e-8) - (
            1 - probs_per_dim
        ) * torch.log(1 - probs_per_dim + 1e-8)
        return persample_entropy.sum(-1).mean()

    def codes_to_indexes(self, zhat):
        if self.basis.numel() != self.embed_dim:
            raise ValueError("Full BSQ indices are unavailable when token_size > 63.")
        if zhat.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}")
        bits = (zhat > 0).to(torch.long)
        return (bits * self.basis).sum(dim=-1).to(torch.long)

    def codes_to_group_indexes(self, zhat):
        zhat_in_group = rearrange(zhat, "b ... (g c) -> b ... g c", c=self.group_size)
        bits = (zhat_in_group > 0).to(torch.long)
        return (bits * self.group_basis).sum(dim=-1).to(torch.long)

    def indexes_to_codes(self, indices):
        if self.basis.numel() != self.embed_dim:
            raise ValueError("Full BSQ codebook lookup is unavailable when token_size > 63.")
        indices = indices.to(device=self.basis.device, dtype=torch.long).unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self.basis), 2)
        return codes_non_centered.to(torch.float32) * 2 - 1

    def group_indexes_to_codes(self, group_indices):
        group_indices = group_indices.to(device=self.group_basis.device, dtype=torch.long).unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(group_indices, self.group_basis), 2)
        codes_non_centered = rearrange(codes_non_centered, "b ... g c -> b ... (g c)")
        return codes_non_centered.to(torch.float32) * 2 - 1

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        return -(probs * torch.log(probs + 1e-8)).sum(dim=dim)

    def get_group_codebook_entry(self, group_indices):
        z_q = self.group_indexes_to_codes(group_indices)
        if self.l2_norm:
            z_q = z_q * (self.embed_dim ** -0.5)
        if self.input_format == "bchw" and group_indices.ndim == 4:
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        return z_q

    def get_codebook_entry(self, indices):
        z_q = self.indexes_to_codes(indices)
        if self.l2_norm:
            z_q = z_q * (self.embed_dim ** -0.5)
        if self.input_format == "bchw" and indices.ndim == 3:
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        return z_q


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.model.image_size
        token_size = config.model.token_size
        vit_enc_patch_size = config.model.vit_enc_patch_size
        vit_enc_model_size = config.model.vit_enc_model_size
        quantizer_mode = config.model.quantize_mode

        add_cls_token = getattr(config.model, 'add_cls_token', False)
        drop_path_rate = getattr(config.model, 'drop_path_rate', 0.)
        mlp_ratio = getattr(config.model, 'mlp_ratio', 4.0)
        qkv_bias = getattr(config.model, 'qkv_bias', True)
        qk_scale = getattr(config.model, 'qk_scale', None)
        qk_norm = getattr(config.model, 'qk_norm', False)
        proj_bias = getattr(config.model, 'proj_bias', True)
        drop_rate = getattr(config.model, 'drop_rate', 0.0)
        attn_drop_rate = getattr(config.model, 'attn_drop_rate', 0.0)
        attn_mode = getattr(config.model, 'attn_mode', 'normal')
        attn_causal = getattr(config.model, 'attn_causal', False)
        gate_attention = getattr(config.model, 'gate_attention', 'none')

        self.image_size = image_size
        self.patch_size = vit_enc_patch_size
        self.model_size = vit_enc_model_size
        self.token_size = token_size

        self.num_image_tokens = image_size[0] * (image_size[1] // self.patch_size)
        self.image_size_after_patch = (image_size[0], image_size[1] // self.patch_size)

        if quantizer_mode == "vae":
            self.token_size = self.token_size * 2  # split into mean and logvar

        self.width = {
            "micro": 192, "tiny": 384, "small": 512,
            "base": 768,  "large": 1024, "huge": 1280, "giant": 1408,
        }[self.model_size]
        self.num_layers = {
            "micro": 4, "tiny": 6,  "small": 8,
            "base": 12, "large": 24, "huge": 32, "giant": 40,
        }[self.model_size]
        self.num_heads = {
            "micro": 3, "tiny": 4,  "small": 8,
            "base": 12, "large": 16, "huge": 16, "giant": 16,
        }[self.model_size]

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=self.patch_size, in_chans=1, embed_dim=self.width)

        scale = self.width ** -0.5
        if add_cls_token:
            self.cls_embed = nn.Parameter(scale * torch.randn(1, self.width))
            self.pos_embed = nn.Parameter(torch.randn(1, 1 + image_size[0], self.width))
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, image_size[0], self.width))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.width, num_heads=self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                qk_norm=qk_norm, proj_bias=proj_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6), attn_mode=attn_mode, attn_causal=attn_causal, gate_attention=gate_attention)
            for i in range(self.num_layers)])

        self.ln_post = nn.LayerNorm(self.width, eps=1e-6)

        # Project width → token_size (or 2*token_size for VAE)
        # self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)
        self.conv_out = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.token_size)
        )

        self.fix_init_weight()

    @property
    def num_patches(self):
        return self.num_image_tokens

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.data.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, need_weights=False):
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Prepend CLS token and add positional embeddings
        if hasattr(self, 'cls_embed'):
            x = torch.cat([_expand_token(self.cls_embed, B).to(x.dtype), x], dim=1)
            pos_embed = self.pos_embed[:, 1:, :]
            num_time_patches = N // pos_embed.shape[1]
            pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, num_time_patches, 1)  # [1, N, T, D]
            pos_embed = pos_embed.reshape(1, -1, D)
            pos_embed = torch.cat([self.pos_embed[:, :1, :], pos_embed], dim=1)  # [1, 1+N*T, D]
            x = x + pos_embed
        else:
            pos_embed = self.pos_embed
            num_time_patches = N // pos_embed.shape[1]
            pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, num_time_patches, 1)  # [1, N, T, D]
            pos_embed = pos_embed.reshape(1, -1, D)
            x = x + pos_embed

        all_attn_weights = [] if need_weights else None
        for block in self.blocks:
            out = block(x, return_attention=need_weights)
            if need_weights:
                x, attn_w = out
                all_attn_weights.append(attn_w)
            else:
                x = out

        if hasattr(self, 'cls_embed'):
            x = x[:, 1:]  # remove CLS token -> (B, num_patches, width)
        x = self.ln_post(x)

        # features: (B, width, num_patches) — pre-conv, used for alignment/downstream
        features = x.permute(0, 2, 1)

        # Reshape to 2D spatial layout and project to token_size
        # x_2d = features.reshape(B, self.width, self.image_size_after_patch[0], self.image_size_after_patch[1])
        # z = self.conv_out(x_2d)  # (B, token_size, H_patches, W_patches)

        z = self.conv_out(features.permute(0, 2, 1))
        z = rearrange(z, 'b (v t) c -> b c v t', v=self.image_size_after_patch[0], t=self.image_size_after_patch[1])

        if need_weights:
            return features, z, all_attn_weights
        return features, z

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = config.model.image_size
        token_size = config.model.token_size
        vit_dec_patch_size = config.model.vit_dec_patch_size
        vit_dec_model_size = config.model.vit_dec_model_size

        add_cls_token = getattr(config.model, 'add_cls_token', False)
        drop_path_rate = getattr(config.model, 'drop_path_rate', 0.)
        mlp_ratio = getattr(config.model, 'mlp_ratio', 4.0)
        qkv_bias = getattr(config.model, 'qkv_bias', True)
        qk_scale = getattr(config.model, 'qk_scale', None)
        qk_norm = getattr(config.model, 'qk_norm', False)
        proj_bias = getattr(config.model, 'proj_bias', True)
        drop_rate = getattr(config.model, 'drop_rate', 0.0)
        attn_drop_rate = getattr(config.model, 'attn_drop_rate', 0.0)
        attn_mode = getattr(config.model, 'attn_mode', 'normal')
        attn_causal = getattr(config.model, 'attn_causal', False)
        gate_attention = getattr(config.model, 'gate_attention', 'none')

        self.image_size = image_size
        self.patch_size = vit_dec_patch_size
        self.model_size = vit_dec_model_size
        self.token_size = token_size

        self.num_image_tokens = image_size[0] * (image_size[1] // self.patch_size)
        self.image_size_after_patch = (image_size[0], image_size[1] // self.patch_size)

        self.width = {
            "micro": 192, "tiny": 384, "small": 512,
            "base": 768,  "large": 1024, "huge": 1280, "giant": 1408,
        }[self.model_size]
        self.num_layers = {
            "micro": 4, "tiny": 6,  "small": 8,
            "base": 12, "large": 24, "huge": 32, "giant": 40,
        }[self.model_size]
        self.num_heads = {
            "micro": 3, "tiny": 4,  "small": 8,
            "base": 12, "large": 16, "huge": 16, "giant": 16,
        }[self.model_size]

        # Project token_size -> width (encoder had width -> token_size)
        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)

        scale = self.width ** -0.5
        if add_cls_token:
            self.cls_embed = nn.Parameter(scale * torch.randn(1, self.width))
            self.pos_embed = nn.Parameter(torch.randn(1, 1 + image_size[0], self.width))
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, image_size[0], self.width))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.width, num_heads=self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                qk_norm=qk_norm, proj_bias=proj_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=nn.LayerNorm, attn_mode=attn_mode, attn_causal=attn_causal, gate_attention=gate_attention)
            for i in range(self.num_layers)])

        self.ln_post = nn.LayerNorm(self.width)

        # self.ffn = nn.Sequential(
        #     nn.Conv2d(self.width, self.patch_size, 1, padding=0, bias=True),
        #     Rearrange('b (p1 c) v t -> b c v (t p1)', p1=self.patch_size),
        # )
        # self.conv_out = nn.Conv2d(1, 1, kernel_size=(1, 3), padding=(0, 1), bias=True)
        self.conv_out = nn.Sequential(
            nn.Linear(self.width * self.image_size_after_patch[1], self.width),
            nn.Tanh(),
            nn.Linear(self.width, image_size[1])
        )

        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, z_quantized, need_weights=False):
        B, C, H_p, W_p = z_quantized.shape
        assert H_p == self.image_size_after_patch[0] and W_p == self.image_size_after_patch[1], (
            f"Expected z_quantized spatial dims {self.image_size_after_patch}, got ({H_p}, {W_p})"
        )

        # Flatten spatial dims: (B, token_size, H_p, W_p) -> (B, num_patches, token_size)
        x = z_quantized.reshape(B, C, H_p * W_p).permute(0, 2, 1)
        x = self.decoder_embed(x)  # (B, num_patches, width)

        if hasattr(self, 'cls_embed'):
            x = torch.cat([_expand_token(self.cls_embed, B).to(x.dtype), x], dim=1)
            pos_embed = self.pos_embed[:, 1:, :]
            num_time_patches = (H_p * W_p) // pos_embed.shape[1]
            pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, num_time_patches, 1)  # [1, N, T, D]
            pos_embed = pos_embed.reshape(1, -1, self.width)
            pos_embed = torch.cat([self.pos_embed[:, :1, :], pos_embed], dim=1)  # [1, 1+N*T, D]
            x = x + pos_embed
        else:
            pos_embed = self.pos_embed
            num_time_patches = (H_p * W_p) // pos_embed.shape[1]
            pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, num_time_patches, 1)  # [1, N, T, D]
            pos_embed = pos_embed.reshape(1, -1, self.width)
            x = x + pos_embed

        all_attn_weights = [] if need_weights else None
        for block in self.blocks:
            out = block(x, return_attention=need_weights)
            if need_weights:
                x, attn_w = out
                all_attn_weights.append(attn_w)
            else:
                x = out

        if hasattr(self, 'cls_embed'):
            x = x[:, 1:]             # remove CLS token -> (B, num_patches, width)
        x = self.ln_post(x)

        # Unpatchify: (B, num_patches, width) -> (B, width, H_p, W_p) -> (B, H, W)
        # x = x.permute(0, 2, 1).reshape(B, self.width, H_p, W_p)
        # x = self.ffn(x.contiguous())
        # x = self.conv_out(x).squeeze(1)  # (B, H, W)
        
        x = rearrange(x, 'b (v t) c -> b v (t c)', v=self.image_size_after_patch[0])
        x = self.conv_out(x)

        if need_weights:
            return x, all_attn_weights
        return x
    
class Tokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config

        self.quantize_mode = str(config.model.quantize_mode).lower()
        try:
            config.model.quantize_mode = self.quantize_mode
        except Exception:
            pass
        if self.quantize_mode not in ["vq", "vae", "fsq", "bsq", "npq"]:
            raise ValueError(f"Unsupported quantize mode: {self.quantize_mode}")

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

        self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.codebook_size,
                token_size=config.model.token_size,
                commitment_cost=config.model.commitment_cost,
                use_l2_norm=config.model.use_l2_norm,
            )
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        elif self.quantize_mode == "fsq":
            fsq_levels = OmegaConf.select(config, "model.fsq_levels")
            if fsq_levels is None:
                fsq_levels = OmegaConf.select(config, "model.levels")
            if fsq_levels is None:
                fsq_levels = [7, 5, 5, 5, 5]
            if OmegaConf.is_config(fsq_levels):
                fsq_levels = OmegaConf.to_container(fsq_levels, resolve=True)
            fsq_levels = [int(level) for level in fsq_levels]

            fsq_num_codebooks = OmegaConf.select(config, "model.fsq_num_codebooks")
            if fsq_num_codebooks is None:
                fsq_num_codebooks = OmegaConf.select(config, "model.num_codebooks")
            fsq_num_codebooks = int(_default(fsq_num_codebooks, 1))

            fsq_keep_num_codebooks_dim = OmegaConf.select(config, "model.fsq_keep_num_codebooks_dim")
            if fsq_keep_num_codebooks_dim is None:
                fsq_keep_num_codebooks_dim = OmegaConf.select(config, "model.keep_num_codebooks_dim")

            self.quantize = FSQ(
                levels=fsq_levels,
                dim=config.model.token_size,
                num_codebooks=fsq_num_codebooks,
                keep_num_codebooks_dim=fsq_keep_num_codebooks_dim,
                scale=OmegaConf.select(config, "model.fsq_scale"),
            )
        elif self.quantize_mode in ("bsq", "npq"):
            bsq_group_size = OmegaConf.select(config, "model.bsq_group_size")
            if bsq_group_size is None:
                bsq_group_size = OmegaConf.select(config, "model.embed_group_size")

            bsq_beta = OmegaConf.select(config, "model.bsq_beta")
            if bsq_beta is None:
                bsq_beta = OmegaConf.select(config, "model.beta")

            bsq_l2_norm = OmegaConf.select(config, "model.bsq_l2_norm")
            if bsq_l2_norm is None:
                bsq_l2_norm = OmegaConf.select(config, "model.post_q_l2_norm")
            if bsq_l2_norm is None:
                bsq_l2_norm = OmegaConf.select(config, "model.use_l2_norm")

            self.quantize = BinarySphericalQuantizer(
                embed_dim=int(config.model.token_size),
                beta=float(_default(bsq_beta, 0.0)),
                gamma0=float(_default(OmegaConf.select(config, "model.bsq_gamma0"), OmegaConf.select(config, "model.gamma0"), 1.0)),
                gamma=float(_default(OmegaConf.select(config, "model.bsq_gamma"), OmegaConf.select(config, "model.gamma"), 1.0)),
                zeta=float(_default(OmegaConf.select(config, "model.bsq_zeta"), OmegaConf.select(config, "model.zeta"), 1.0)),
                input_format="bchw",
                soft_entropy=_as_bool(_default(OmegaConf.select(config, "model.bsq_soft_entropy"), OmegaConf.select(config, "model.soft_entropy"), True)),
                group_size=int(_default(bsq_group_size, 1)),
                index_mode=str(_default(
                    OmegaConf.select(config, "model.bsq_index_mode"),
                    OmegaConf.select(config, "model.index_mode"),
                    "full",
                )),
                persample_entropy_compute=str(_default(
                    OmegaConf.select(config, "model.bsq_persample_entropy_compute"),
                    OmegaConf.select(config, "model.persample_entropy_compute"),
                    "group",
                )),
                cb_entropy_compute=str(_default(
                    OmegaConf.select(config, "model.bsq_cb_entropy_compute"),
                    OmegaConf.select(config, "model.cb_entropy_compute"),
                    "group",
                )),
                l2_norm=_as_bool(_default(bsq_l2_norm, False)),
                inv_temperature=float(_default(
                    OmegaConf.select(config, "model.bsq_inv_temperature"),
                    OmegaConf.select(config, "model.inv_temperature"),
                    1.0,
                )),
            )
        else:
            raise NotImplementedError

        self.loss_fn = F.mse_loss
        self.kl_weight = getattr(config.model, 'kl_weight', 1e-6)
        self.quantizer_weight = getattr(config.model, 'quantizer_weight', 1.0)

    @property
    def codebook_size(self):
        """Number of codebook entries; 0 for non-discrete modes."""
        return self.quantize.codebook_size if self.quantize_mode in ("vq", "fsq", "bsq", "npq") else 0

    def get_codebook_indices(self, x):
        """Return flat codebook indices (B, num_patches) for the fMRI-NTP objective.

        Valid in VQ, FSQ, BSQ, and NPQ modes.
        """
        assert self.quantize_mode in ("vq", "fsq", "bsq", "npq"), "get_codebook_indices is only valid in VQ/FSQ/BSQ/NPQ modes"
        _, _, _, result_dict = self.encode(x)
        indices = result_dict['min_encoding_indices']
        return indices.reshape(indices.shape[0], -1)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        """Encode input to quantized patch codes.

        Args:
            x: (B, H, W) or (B, 1, H, W)

        Returns:
            features:    (B, width, num_patches)  — pre-conv encoder features
            z:           (B, token_size, H_patches, W_patches)  — post-conv encoder output
            z_quantized: (B, token_size, H_patches, W_patches)
            result_dict: quantizer metadata dict or DiagonalGaussianDistribution object
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # B 1 H W

        features, z = self.encoder(x)
        if self.quantize_mode == "vq":
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "vae":
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            result_dict = posteriors
        elif self.quantize_mode == "fsq":
            z_quantized, min_encoding_indices = self.quantize(z)
            result_dict = {
                'quantizer_loss': z.new_zeros(()),
                'min_encoding_indices': min_encoding_indices,
            }
        elif self.quantize_mode in ("bsq", "npq"):
            z_quantized, result_dict = self.quantize(z)
        else:
            raise NotImplementedError
        return features, z, z_quantized, result_dict

    def decode(self, z_quantized):
        """Decode quantized patch codes back to signal space.

        Args:
            z_quantized: (B, token_size, H_patches, W_patches)

        Returns:
            decoded: (B, H, W)
        """
        return self.decoder(z_quantized)

    def forward_encoder(self, x, wo_conv=False):
        """Forward pass through encoder only (no quantization).

        Args:
            x: (B, H, W) or (B, 1, H, W)
            wo_conv: if True return pre-conv features (B, width, num_patches),
                     otherwise return post-conv z (B, token_size, H_p, W_p)

        Returns:
            features or z depending on wo_conv
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        features, z = self.encoder(x)
        return features if wo_conv else z

    def forward(self, x, y=None, encoding_only=False, wo_conv=False):
        """Full encode-quantize-decode forward pass.

        Args:
            x: (B, H, W) or (B, 1, H, W)
            y: optional labels for conditional processing
            wo_conv: passed to encoder when encoding_only=True

        Returns:
            (decoded, result_dict)  or  z  (when encoding_only=True)
        """
        if encoding_only:
            return self.forward_encoder(x, wo_conv=wo_conv)

        features, z, z_quantized, result_dict = self.encode(x)
        decoded = self.decode(z_quantized)

        if y is None:
            return None, features, None

        rec_loss = self.loss_fn(decoded, y)

        if self.quantize_mode == 'vae':
            kl_loss = result_dict.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss = rec_loss + self.kl_weight * kl_loss
            quant_loss = kl_loss
        else:
            quantizer_loss = result_dict['quantizer_loss']
            loss = rec_loss + self.quantizer_weight * quantizer_loss
            quant_loss = quantizer_loss

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = quant_loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, features, log
    
class TokAlign(nn.Module):
    def __init__(self, config, lm_name='gpt2', **Kwargs):
        super().__init__()
        self.config = config
        self.lm_name = lm_name
        
        self.quantizer = Tokenizer(config)
        self.num_tokens = self.quantizer.encoder.num_image_tokens
        self.enc_embed_dim = self.quantizer.encoder.width

        model_hf = AutoModelForCausalLM.from_pretrained(lm_name)
        self.hidden_dim = model_hf.config.hidden_size

        self.x_proj = nn.Linear(self.enc_embed_dim, self.hidden_dim)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )

        self.wte = model_hf.get_input_embeddings()
        for p in self.wte.parameters():
            p.requires_grad = False
        
        self.domain_classifier.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y=None, alpha=0):
        if y is not None:
            # Single encoder pass: features (B, width, num_patches) returned alongside loss
            loss, features, log = self.quantizer(x, y)
            # (B, width, num_patches) -> (B, num_patches, width)
            encoder_features = features.permute(0, 2, 1)
            encoder_features = self.x_proj(encoder_features)  # (B, num_patches, hidden_dim)
            reverse_x = ReverseLayerF.apply(encoder_features, alpha)
            domain_out = self.domain_classifier(reverse_x)  # (B, num_patches, 2)
            # fMRI = domain 0
            target = torch.zeros(domain_out.size(0) * domain_out.size(1), dtype=torch.long, device=x.device)
            domain_loss = F.cross_entropy(domain_out.view(-1, 2), target)
            split = "train" if self.training else "val"
            log[f'{split}/domain_loss'] = domain_loss.detach().item()
            return loss, domain_loss, log
        else:
            with torch.no_grad():
                x = self.wte(x).detach()  # (B, seq_len, hidden_dim)
            domain_out = self.domain_classifier(x)  # (B, seq_len, 2)
            # text = domain 1
            target = torch.ones(x.size(0) * x.size(1), dtype=torch.long, device=x.device)
            domain_loss = F.cross_entropy(domain_out.view(-1, 2), target)
            return domain_loss
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters 
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


if __name__ == "__main__":
    # ── Minimal smoke test for Tokenizer (VQ, VAE, and FSQ modes) ─────────────
    import sys

    # Small spatial dims so the test runs quickly on CPU:
    #   image_size = (ROIs, timepoints), patch_size divides timepoints
    #   here: 8 ROIs × 32 timepoints, patch_size=16  →  W_patches=2, H_patches=8
    IMAGE_SIZE = (450, 160)
    PATCH_SIZE = 32
    TOKEN_SIZE = 128
    B = 2  # batch size

    def make_config(quantize_mode):
        return OmegaConf.create({
            "model": {
                "quantize_mode": quantize_mode,
                "image_size": list(IMAGE_SIZE),
                "token_size": TOKEN_SIZE,
                "vit_enc_patch_size": PATCH_SIZE,
                "vit_enc_model_size": "micro",   # 4 layers, width=192
                "vit_dec_patch_size": PATCH_SIZE,
                "vit_dec_model_size": "micro",
                # VQ-specific
                "codebook_size": 64,
                "commitment_cost": 0.25,
                "use_l2_norm": True,
                # FSQ-specific
                "fsq_levels": [7, 5, 5, 5, 5],
                "fsq_num_codebooks": 1,
                # loss weights
                "kl_weight": 1e-6,
                "quantizer_weight": 1.0,
            }
        })

    def test_tokenizer(quantize_mode):
        print(f"\n{'='*50}")
        print(f"Testing Tokenizer  [quantize_mode={quantize_mode}]")
        cfg = make_config(quantize_mode)
        model = Tokenizer(cfg)
        model.eval()

        x = torch.randn(B, *IMAGE_SIZE)   # (B, H, W)
        y = torch.randn(B, *IMAGE_SIZE)   # reconstruction target

        # ── encode ──
        features, z_pre, z_q, result = model.encode(x)
        print(f"  encode features : {tuple(features.shape)}")
        print(f"  encode   z_pre  : {tuple(z_pre.shape)}")
        print(f"  encode   z_q    : {tuple(z_q.shape)}")

        # ── decode ──
        recon = model.decode(z_q)
        print(f"  decode   recon : {tuple(recon.shape)}")
        assert recon.shape == x.shape, f"Reconstruction shape mismatch: {recon.shape} vs {x.shape}"

        # ── full forward (with target) ──
        loss, z_raw, log = model(x, y=y)
        assert loss is not None and loss.ndim == 0, "Expected scalar loss"
        print(f"  forward  loss  : {loss.item():.4f}")
        print(f"  log keys : {list(log.keys())}")

        # ── encoding_only ──
        z_enc = model(x, encoding_only=True)
        print(f"  encoding_only  : {tuple(z_enc.shape)}")

        print(f"  PASSED ✓")

    ok = True
    for mode in ("vq", "vae", "fsq"):
        try:
            test_tokenizer(mode)
        except Exception as e:
            print(f"  FAILED ✗  {e}", file=sys.stderr)
            ok = False

    # ── TokAlign test ────────────────────────────────────────────────────────
    def test_tokalign():
        print(f"\n{'='*50}")
        print("Testing TokAlign")
        cfg = make_config("vq")
        model = TokAlign(cfg, lm_name='gpt2')
        model.eval()

        num_patches = model.num_tokens  # H_patches * W_patches = ROIs * (T // patch_size)
        hidden_dim  = model.hidden_dim

        # ── fMRI branch (y is not None): returns (loss, domain_loss, log) ──
        x_fmri = torch.randn(B, *IMAGE_SIZE)
        y_fmri = torch.randn(B, *IMAGE_SIZE)
        loss, domain_loss, log = model(x_fmri, y=y_fmri, alpha=1.0)
        assert loss is not None and loss.ndim == 0,        "fMRI: expected scalar rec loss"
        assert domain_loss.ndim == 0,                      "fMRI: expected scalar domain loss"
        split = "val"
        assert f"{split}/domain_loss" in log,              "fMRI: domain_loss missing from log"
        print(f"  fMRI   rec_loss={loss.item():.4f}  domain_loss={domain_loss.item():.4f}")

        # ── text branch (y is None): returns domain_loss only ──
        SEQ_LEN = 16
        # wte vocab size
        vocab_size = model.wte.weight.shape[0]
        x_text = torch.randint(0, vocab_size, (B, SEQ_LEN))
        domain_loss_text = model(x_text, y=None)
        assert domain_loss_text.ndim == 0, "text: expected scalar domain loss"
        print(f"  text   domain_loss={domain_loss_text.item():.4f}")

        print("  PASSED ✓")

    try:
        test_tokalign()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAILED ✗  {e}", file=sys.stderr)
        ok = False

    sys.exit(0 if ok else 1)
