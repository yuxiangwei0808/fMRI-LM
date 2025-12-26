import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(450, 490), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0]) * (img_size[1] // patch_size)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_time_patches = img_size[1] // patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        B, C, N, T = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        # TODO test patch embed similar to NeuroLM
        # x = rearrange(self.proj(x), 'b c n t -> b n (t c)')
        return x