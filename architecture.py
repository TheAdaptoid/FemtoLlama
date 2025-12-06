from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-8):
        """
        Initialize the RMSNorm layer.

        Args:
            d_model (int): The dimension of the model.
            eps (float): A small value to avoid division by zero.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the RMSNorm of the input tensor."""
        mean_square = x.pow(2).mean(-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        x_normed = x / rms
        return x_normed * self.scale


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, d_model: int, max_seq_len: int):
        """
        Initialize the RoPE layer.

        Args:
            d_model (int): The dimension of the model.
            max_seq_len (int): Maximum sequence length.
        """
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embeddings."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]

        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        return x * cos_emb + x_rot * sin_emb


class SelfAttention(nn.Module):
    """Multi-Head Self Attention with RoPE."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int):
        """
        Initialize the Self Attention layer.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            max_seq_len (int): Maximum sequence length.
        """
        super(SelfAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        # Calculate dimension per attention head
        self.head_dim = d_model // num_heads

        # Linear projections for query, key, value, and output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Initialize RoPE for positional embeddings
        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute multi-head self attention."""
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape

        # Project input and reshape for multi-head attention
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply rotary position embeddings to queries and keys
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Reshape context back to original dimensions
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        # Project output
        output = self.out_proj(context)

        return output


class SwiGLU(nn.Module):
    """SwiGLU Activation Function."""

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the SwiGLU layer.

        Args:
            input_dim (int): The dimension of the input.
            output_dim (int): The dimension of the output.
        """
        super(SwiGLU, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SwiGLU activation."""
        values, gates = self.proj(x).chunk(2, dim=-1)
        return F.silu(values) * gates


class FeedForward(nn.Module):
    """
    Two-Layer Feed Forward Network.

    The first layer is a SwiGLU activated linear layer,
    followed by a linear output layer.
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize the Feed Forward layer.

        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feed forward layer.
        """
        super(FeedForward, self).__init__()
        self.swi_glu = SwiGLU(d_model, d_ff)
        self.out_layer = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the feed forward network."""
        return self.out_layer(self.swi_glu(x))


class TransformerBlock(nn.Module):
    """Transformer Block with RMSNorm, Self-Attention, and a Feed-Forward Network."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int):
        """
        Initialize the Transformer Block.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            max_seq_len (int): Maximum sequence length.
        """
        super(TransformerBlock, self).__init__()

        # The original LLaMA used a factor of 8/3.
        d_hidden: int = floor((8 / 3) * d_model)

        self.rms_norm1 = RMSNorm(d_model)
        self.self_attention = SelfAttention(d_model, num_heads, max_seq_len)
        self.rms_norm2 = RMSNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_hidden)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the Transformer Block."""
        # Pre-normalization
        x_norm_1 = self.rms_norm1(x)
        # Self-Attention
        attn_output = self.self_attention(x_norm_1, mask)
        # Residual connection
        x = x + attn_output

        # Pre-normalization
        x_norm_2 = self.rms_norm2(x)
        # Feed-Forward Network
        ff_output = self.feed_forward(x_norm_2)
        # Residual connection
        x = x + ff_output

        return x


class LM_Head(nn.Module):
    """Language Model Head for token prediction."""

    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the Language Model Head.

        Args:
            d_model (int): The dimension of the model.
            vocab_size (int): The size of the vocabulary.
        """
        super(LM_Head, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the logits for token prediction."""
        return self.linear(x)


class FemtoLlama(nn.Module):
    """FemtoLlama Model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        context_window_len: int,
    ):
        """
        Initialize the FemtoLlama model.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
            context_window_len (int): Maximum sequence length.
        """
        super(FemtoLlama, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, context_window_len)
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model)
        self.lm_head = LM_Head(d_model, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the FemtoLlama model output."""
        x = self.token_embedding(input_ids)

        # Move forward through each transformer layer
        for layer in self.layers:
            x = layer(x, mask)

        x = self.rms_norm(x)
        logits = self.lm_head(x)

        return logits
