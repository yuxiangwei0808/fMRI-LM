# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# MAE-ST: https://github.com/facebookresearch/mae_st
# BrainJEPA: https://github.com/Eric-LRL/Brain-JEPA
# --------------------------------------------------------

import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

from .vision_transformer import VisionTransformer, Block
from .patch_embed import PatchEmbed

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone for fMRI data (V x T)"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=False,
        attn_mode='normal',
        gradient_checkpointing=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.init_std = init_std
        self.in_chans = in_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.patch_size = patch_size
        self.img_size = img_size

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        
        # Calculate number of spatial and temporal patches
        self.num_spatial_patches = img_size[0]  # V dimension
        self.num_temporal_patches = img_size[1] // patch_size  # T dimension patchified

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        if sep_pos_embed:
            # Separate spatial and temporal positional embeddings
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_patches, embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patches, embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            # Single positional embedding for all patches
            if self.cls_embed:
                # _num_patches = self.num_patches + 1
                _num_patches = img_size[0] + 1
            else:
                # _num_patches = self.num_patches
                _num_patches = img_size[0]
            self.pos_embed = nn.Parameter(torch.zeros(1, _num_patches, embed_dim))

        # Encoder blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                attn_mode=attn_mode
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if self.cls_embed:
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embeddings
        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_patches, decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patches, decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                # _num_patches = self.num_patches + 1
                _num_patches = img_size[0] + 1
            else:
                # _num_patches = self.num_patches
                _num_patches = img_size[0]
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim)
            )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.0,  # No drop path in decoder
                norm_layer=norm_layer,
                attn_mode=attn_mode
            )
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Decoder prediction head: reconstruct patch_size values per patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * in_chans,
            bias=True
        )

        self.norm_pix_loss = norm_pix_loss

        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize cls tokens
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_cls_token, std=0.02)

        # Initialize positional embeddings
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)
            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w, std=0.02)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        
        # Apply rescaling to attention and MLP layers
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        
        for layer_id, layer in enumerate(self.decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight.view([m.weight.shape[0], -1]))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: (B, C, V, T) where V is spatial, T is temporal
        x: (B, V*T_patches, patch_size*C) where T_patches = T // patch_size
        """
        if imgs.dim() == 3: imgs = imgs.unsqueeze(1)
        B, C, V, T = imgs.shape
        p = self.patch_size
        assert T % p == 0, f"Temporal dimension {T} must be divisible by patch_size {p}"
        
        T_patches = T // p
        
        # Reshape: (B, C, V, T_patches, p)
        x = imgs.reshape(B, C, V, T_patches, p)
        # Transpose: (B, V, T_patches, p, C)
        x = x.permute(0, 2, 3, 4, 1)
        # Flatten: (B, V*T_patches, p*C)
        x = x.reshape(B, V * T_patches, p * C)
        
        return x

    def unpatchify(self, x):
        """
        x: (B, V*T_patches, patch_size*C)
        imgs: (B, C, V, T)
        """
        B = x.shape[0]
        V = self.num_spatial_patches
        T_patches = self.num_temporal_patches
        p = self.patch_size
        C = self.in_chans
        
        # Reshape: (B, V, T_patches, p, C)
        x = x.reshape(B, V, T_patches, p, C)
        # Transpose: (B, C, V, T_patches, p)
        x = x.permute(0, 4, 1, 2, 3)
        # Flatten temporal: (B, C, V, T)
        imgs = x.reshape(B, C, V, T_patches * p)
        
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # Embed patches
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Add pos embed (before masking)
        if self.sep_pos_embed:
            # Compute spatial and temporal positional embeddings
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.num_temporal_patches, 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patches,
                dim=1,
            )
        else:
            if self.cls_embed:
                pos_embed = self.pos_embed[:, 1:, :]
            else:
                pos_embed = self.pos_embed
            num_time_patches = N // pos_embed.shape[1]
            pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, num_time_patches, 1)  # [1, N, T, D]
            pos_embed = pos_embed.reshape(1, -1, D)                
        x = x + pos_embed

        # Masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        # Append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            if self.sep_pos_embed:
                cls_token = cls_token + self.pos_embed_class
            else:
                cls_token = cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)

        # Remove cls token before decoder
        if self.cls_embed:
            x = x[:, 1:, :]

        return x, mask, ids_restore

    def forward_encoder_nonmask(self, x):
        # Embed patches
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Add pos embed
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.num_temporal_patches, 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patches,
                dim=1,
            )
        else:
            if self.cls_embed:
                pos_embed = self.pos_embed[:, 1:, :]
            else:
                pos_embed = self.pos_embed
            num_time_patches = N // pos_embed.shape[1]
            pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, num_time_patches, 1)  # [1, N, T, D]
            pos_embed = pos_embed.reshape(1, -1, D)

        x = x + pos_embed

        # Apply Transformer blocks
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)

        return x

    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        B, _, C = x.shape

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, self.num_patches - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )  # unshuffle
        x = x_

        # Append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(B, -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        # Add pos embed
        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.num_temporal_patches, 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.num_spatial_patches,
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class,
                        decoder_pos_embed,
                    ],
                    dim=1,
                )
        else:
            # Broadcast spatial pos_embed over time patches (matches forward_encoder)
            num_time_patches = self.num_patches // self.decoder_pos_embed.shape[1]
            decoder_pos_embed = self.decoder_pos_embed.unsqueeze(2).repeat(
                1, 1, num_time_patches, 1
            )
            decoder_pos_embed = decoder_pos_embed.reshape(1, -1, C)
            if self.cls_embed:
                # prepend a zero cls pos embed (decoder cls token has no dedicated pos)
                cls_pos = torch.zeros(1, 1, C, device=x.device, dtype=x.dtype)
                decoder_pos_embed = torch.cat([cls_pos, decoder_pos_embed], dim=1)

        x = x + decoder_pos_embed

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls token
        if self.cls_embed:
            x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, V, T]
        pred: [N, V*T_patches, patch_size*C]
        mask: [N, V*T_patches], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.5, encoding_only=True):
        """
        imgs: [N, C, V, T] - fMRI data with shape (batch, channels, spatial, temporal)
        mask_ratio: ratio of patches to mask
        """
        weight_dtype = self.pos_embed.dtype if not self.sep_pos_embed else self.pos_embed_spatial.dtype
        if imgs.dtype != weight_dtype:
            imgs = imgs.to(weight_dtype)
        if encoding_only:
            return self.forward_encoder_nonmask(imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)

        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        # return loss, pred, mask
        return latent, loss


def mae_vit_small(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=8,
        decoder_num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs,
    )
    return model


def mae_vit_base(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


if __name__ == '__main__':
    # Example usage
    model = mae_vit_small(
        img_size=(450, 160),  # V=450, T=160 (divisible by patch_size=16)
        patch_size=160,
        in_chans=1,
        sep_pos_embed=True,
        cls_embed=False,
        norm_pix_loss=True
    ).cuda()
    
    x = torch.randn(2, 1, 450, 160).cuda()
    out = model(x, mask_ratio=0.75, encoding_only=False)
    print(out.shape)