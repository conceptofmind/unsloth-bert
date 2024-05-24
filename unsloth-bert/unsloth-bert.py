import torch
import torch.nn as nn
import torch.nn.functional as F

from rmsnorm import Fast_RMS_Layernorm
from rope import fast_rope_embedding
from grad_check_offload import Offloaded_Gradient_Checkpointer

# Unsloth RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        return Fast_RMS_Layernorm.apply(x, self.weight, self.variance_epsilon)


# Unsloth RotaryEmbedding
class RotaryEmbedding(torch.nn.Module):
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())
    pass

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.max_seq_len_cached = seq_len
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float() / self.dim)
        )
        t = torch.arange(self.max_seq_len_cached, device="cpu", dtype=torch.int64).float()

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=dtype, device=device, non_blocking=True), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=dtype, device=device, non_blocking=True), persistent=False)
    pass

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
    pass
pass


# Torch Script SQReLU
@torch.jit.script
def sqrelu_fwd(x):
    r = F.relu(x)
    return (r * r).to(dtype=x.dtype)

@torch.jit.script
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x)).to(dtype=x.dtype)

class SQ_ReLU_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return sqrelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tmp = sqrelu_bwd(grad_output, input)
        return tmp

sq_relu_impl = SQ_ReLU_Function.apply


# MLP
class MLP(nn.Module):
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int, 
        dropout: float,
        use_bias: bool,
    ):
        super().__init__()

        self.linear_in = nn.Linear(dim, hidden_dim, bias=use_bias)

        self.act_fn = sq_relu_impl
        
        self.d1 = nn.Dropout(dropout)
        
        self.linear_out = nn.Linear(hidden_dim, dim, bias=use_bias)
        
        self.d2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.d1(x)
        x = self.linear_out(x)
        x = self.d2(x)
        return x

# Attention
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        use_qkv_bias: bool,
        use_bias: bool,
        scale: float = None,
    ):
        super().__init__()

        head_dim = int(dim // heads)
        inner_dim = int(head_dim * heads) 
        
        self.heads = heads
        self.head_dim = head_dim

        assert int(dim // heads) <= 256 and int(dim // heads) >= 64, "dim // heads must be between 64 and 256."
        assert inner_dim % heads == 0, "num_heads must divide dim."

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=use_qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=use_bias)

        self.rotary_emb = RotaryEmbedding(dim)

        self.dropout = dropout
        self.scale = scale

    def forward(self, x):
        """
        b: batch size
        n: sequence length
        h: number of heads
        head_dim: dim // heads
        """
        b, n, h = x.shape[0], x.shape[1], self.heads

        """
        In Shape: (b, n, dim)
        Out Shape: (b, n, dim)
        """
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        """
        In Shape: (b, n, dim)
        View: (b, n, dim) -> (b, n, h, head_dim)
        Transpose: (b, n, h, head_dim) -> (b, h, n, head_dim)
        Out Shape: (b, h, n, head_dim)
        """
        q = q.view(b, n, h, self.head_dim).transpose(1, 2)
        k = k.view(b, n, h, self.head_dim).transpose(1, 2)
        v = v.view(b, n, h, self.head_dim).transpose(1, 2)

        """
        In Shape: (b, h, n, head_dim)
        """
        cos_cached = self.rotary_emb.cos_cached
        sin_cached = self.rotary_emb.sin_cached
        q, k = fast_rope_embedding(q, k, cos_cached, sin_cached)

        """
        In Shape: (b, h, n, head_dim)
        Out Shape: (b, h, n, head_dim)
        """
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            is_causal=False, 
            dropout_p=self.dropout,
            scale=self.scale,
        )

        """
        In Shape: (b, h, n, head_dim)
        Transpose: (b, h, n, head_dim) -> (b, n, h, head_dim)
        Reshape: (b, n, h, head_dim) -> (b, n, head_dim * h)
        Out Shape: (b, n, head_dim * h)
        """
        out = out.transpose(1, 2).contiguous().reshape(b, n, self.head_dim * h)
        out = self.to_out(out)

        return out


# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        use_qkv_bias: bool,
        use_bias: bool,
    ):
        super().__init__()

        ff_dim = int(dim * 4)

        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(
            dim, 
            heads, 
            dropout,
            use_qkv_bias,
            use_bias,
        )
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MLP(
            dim, 
            ff_dim, 
            dropout, 
            use_bias
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float,
        use_qkv_bias: bool,
        use_bias: bool,
        use_checkpointing: bool,
    ):
        super().__init__()

        self.use_checkpointing = use_checkpointing

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                TransformerBlock(dim, heads, dropout, use_qkv_bias, use_bias,)
            )

    def forward(self, x):
        for block in self.layers:
            if self.use_checkpointing:
                x = Offloaded_Gradient_Checkpointer.apply(block, x)
            else:
                x = block(x)
        return x


# LMHead
class LMHead(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        use_bias: bool,
    ):
        super().__init__()

        self.lmhead_norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=use_bias)

    def forward(self, x):
        x = self.lmhead_norm(x)
        x = self.to_logits(x)
        return x


# Unsloth Bert
class UnslothBERT(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
        use_qkv_bias: bool = True,
        use_bias: bool = True,
        use_checkpointing: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.embed = nn.Embedding(num_tokens, dim, padding_idx,)
        self.transformer = Transformer(
            dim, 
            depth, 
            heads,
            dropout,
            use_qkv_bias,
            use_bias,
            use_checkpointing,
        )
        self.lmhead = LMHead(dim, num_tokens, use_bias,)

        self.lmhead.to_logits.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = self.lmhead(x)
        return x


if __name__ == "__main__":

    # Test Unsloth BERT
    x = torch.randint(0, 32000, (1, 2048)).cuda()

    model = UnslothBERT(
        num_tokens=32000, 
        dim=2048,
        depth=6,
        heads=8,
    ).cuda()

    model = torch.compile(model)

    out = model(x)

    print(out)

    n_params_torch = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Number of parameters in torch model: {n_params_torch}")