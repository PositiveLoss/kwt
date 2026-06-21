"""
kwt.py
This module contains the KWT class, a Tranformer-based model for keyword detection.
"""

# Imports

from typing import Any, NotRequired, TypedDict, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from utils.helion_kernels import HelionGELU


class KWTConfig(TypedDict):
    input_res: list[int]
    patch_res: list[int]
    num_classes: int
    mlp_dim: int
    dim: int
    heads: int
    depth: int
    dropout: float
    emb_dropout: float
    pre_norm: bool
    pool: NotRequired[str]
    channels: NotRequired[int]
    dim_head: NotRequired[int]
    use_sdpa: bool
    use_helion_kernels: bool


# Basically vision transformer, ViT that accepts MFCC + SpecAug. Refer to:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


class PreNorm(nn.Module):
    """
    Pre-normalization module that applies layer normalization before the input is passed to the given function.
    """

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Apply layer normalization to the input tensor and pass it to the given function.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    """
    Post-normalization module that applies layer normalization after the input is passed to the given function.
    """

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Pass the input tensor to the given function and apply layer normalization to the output.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    """
    Feedforward module that applies two linear layers with GELU activation and dropout.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_helion_kernels: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            HelionGELU(enabled=use_helion_kernels),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input tensor through the feedforward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention layer.

    Args:
        dim (int): Input feature dimension.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        dropout (float): Dropout rate.

    Attributes:
        heads (int): Number of attention heads.
        to_qkv (nn.Linear): Linear layer to project inputs to queries, keys, and values.
        to_out (nn.Module): Output projection layer.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.use_sdpa = use_sdpa
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        Args:
            x: Input tensor.

        Returns:
            The output of the transformer model.
        """
        h = self.heads
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=h)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer model.

    Args:
        dim: Input feature dimension.
        depth: Number of transformer layers.
        heads: Number of attention heads.
        dim_head: Dimension of each attention head.
        mlp_dim: Dimension of the intermediate layer in the feedforward network.
        pre_norm: Whether to use pre-normalization or post-normalization.
        dropout: Dropout probability.

    Returns:
        The output of the transformer model.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        pre_norm: bool = True,
        dropout: float = 0.0,
        use_sdpa: bool = True,
        use_helion_kernels: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        P_Norm = PreNorm if pre_norm else PostNorm

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        P_Norm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_sdpa=use_sdpa,
                            ),
                        ),
                        P_Norm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout=dropout,
                                use_helion_kernels=use_helion_kernels,
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        Args:
            x: Input tensor.

        Returns:
            The output of the transformer model.
        """
        for layer in self.layers:
            attn, ff = cast(tuple[nn.Module, nn.Module], tuple(layer.children()))
            x = attn(x) + x
            x = ff(x) + x
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_res: list[int], patch_dim: int, dim: int):
        super().__init__()
        self.patch_height = patch_res[0]
        self.patch_width = patch_res[1]
        self.proj = nn.Linear(patch_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_height,
            p2=self.patch_width,
        )
        return self.proj(x)


class KWT(nn.Module):
    """
    Keyword Transformer (KWT) model.

    Args:
        input_res (list[int]): Input resolution of the spectrogram.
        patch_res (list[int]): Resolution of the patches.
        num_classes (int): Number of classes.
        dim (int): Embedding dimension.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimension of the MLP.
        pool (str): Pooling type, either "cls" (cls token) or "mean" (mean pooling).
        channels (int): Number of input channels.
        dim_head (int): Dimension of each attention head.
        dropout (float): Dropout rate.
        emb_dropout (float): Embedding dropout rate.
        pre_norm (bool): Whether to use pre-normalization.
        use_sdpa (bool): Whether to use PyTorch scaled dot-product attention.
        use_helion_kernels (bool): Whether to use optional Helion kernels when supported.
    """

    def __init__(
        self,
        input_res: list[int],
        patch_res: list[int],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        channels: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        pre_norm: bool = True,
        use_sdpa: bool = True,
        use_helion_kernels: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        if input_res[0] % patch_res[0] or input_res[1] % patch_res[1]:
            raise ValueError("input_res dimensions must be divisible by patch_res.")

        num_patches = (input_res[0] // patch_res[0]) * (input_res[1] // patch_res[1])

        patch_dim = channels * patch_res[0] * patch_res[1]
        if pool not in {"cls", "mean"}:
            raise ValueError("pool type must be either cls or mean.")

        self.to_patch_embedding = PatchEmbedding(patch_res, patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            pre_norm,
            dropout,
            use_sdpa,
            use_helion_kernels,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        Args:
            x: Input tensor.

        Returns:
            The output of the KWT model.
        """
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def kwt_from_name(model_name: str) -> KWT:
    """
    Returns a KWT model based on the specified model name.

    Args:
        model_name: The name of the KWT model to use.

    Returns:
        A KWT model with the specified configuration.

    Raises:
        AssertionError: If the specified model name is not supported.
    """
    models: dict[str, KWTConfig] = {
        "kwt-1": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 256,
            "dim": 64,
            "heads": 1,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False,
            "use_sdpa": True,
            "use_helion_kernels": False,
        },
        "kwt-2": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 512,
            "dim": 128,
            "heads": 2,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False,
            "use_sdpa": True,
            "use_helion_kernels": False,
        },
        "kwt-3": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 768,
            "dim": 192,
            "heads": 3,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False,
            "use_sdpa": True,
            "use_helion_kernels": False,
        },
    }

    if model_name not in models:
        raise ValueError(
            f"Unsupported model_name {model_name}; must be one of {list(models)}."
        )

    return KWT(**models[model_name])
