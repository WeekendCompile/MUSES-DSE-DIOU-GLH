import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class GaussianHistoryBank(nn.Module):
    """
    Hybrid Gaussian Latent History.

    A global learnable action-prototype dictionary (mu_prior, log_var_prior, w_prior)
    is adapted per input segment via data-conditioned deltas computed from a
    temporally-attentive context vector (single-head self-attn pool + mean/max pool).

        mu_k(F_t)      = mu_prior_k       + g * Delta_mu_k(ctx(F_t))
        log_var_k(F_t) = log_var_prior_k  + g * Delta_logvar_k(ctx(F_t))
        w_k(F_t)       = w_prior_k        + g * Delta_w_k(ctx(F_t))

    Sampling uses the reparameterization trick during training; deterministic at eval.
    """
    def __init__(self, num_gaussians, embedding_dim, init_log_var=-2.0):
        super(GaussianHistoryBank, self).__init__()
        self.K = num_gaussians
        self.D = embedding_dim

        # Global prior: action prototype dictionary
        self.mu_prior = nn.Parameter(torch.randn(num_gaussians, embedding_dim) * 0.02)
        self.log_var_prior = nn.Parameter(torch.full((num_gaussians, embedding_dim), init_log_var))
        self.weight_logits_prior = nn.Parameter(torch.zeros(num_gaussians))

        # Temporal context: single-head attention pool captures ordering, not just statistics.
        # Query is the global mean; keys/values are the sequence tokens.
        self.ctx_q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.ctx_k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.ctx_v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.ctx_attn_scale = embedding_dim ** -0.5

        # Fuse attn-pool + mean + max into a single context vector
        self.context_proj = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )

        # Per-input adaptation heads (zero-init so model starts at the prior)
        self.delta_mu_head = nn.Linear(embedding_dim, num_gaussians * embedding_dim)
        self.delta_logvar_head = nn.Linear(embedding_dim, num_gaussians * embedding_dim)
        self.delta_weight_head = nn.Linear(embedding_dim, num_gaussians)
        for layer in (self.delta_mu_head, self.delta_logvar_head, self.delta_weight_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Adaptation gate (sigmoid(-2) ~ 0.12 so prior dominates early in training)
        self.adapt_gate = nn.Parameter(torch.tensor(-2.0))

    def summarize(self, x):
        # x: [T, B, D]
        mean_pool = x.mean(dim=0)                           # [B, D] — global mean as query
        max_pool = x.max(dim=0)[0]                          # [B, D]

        # Single-head attention pool: query=mean, keys/values=sequence
        q = self.ctx_q_proj(mean_pool).unsqueeze(1)         # [B, 1, D]
        k = self.ctx_k_proj(x).permute(1, 0, 2)            # [B, T, D]
        v = self.ctx_v_proj(x).permute(1, 0, 2)            # [B, T, D]
        attn_w = torch.softmax(
            torch.bmm(q, k.transpose(1, 2)) * self.ctx_attn_scale, dim=-1
        )                                                   # [B, 1, T]
        attn_pool = torch.bmm(attn_w, v).squeeze(1)        # [B, D]

        ctx = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)   # [B, 3D]
        return self.context_proj(ctx)                               # [B, D]

    def sample(self, x):
        seq_len, batch, dim = x.shape

        ctx = self.summarize(x)                                              # [B, D]
        delta_mu = self.delta_mu_head(ctx).view(batch, self.K, self.D)       # [B, K, D]
        delta_logvar = self.delta_logvar_head(ctx).view(batch, self.K, self.D)
        delta_weight = self.delta_weight_head(ctx)                           # [B, K]

        gate = torch.sigmoid(self.adapt_gate)
        mu = self.mu_prior.unsqueeze(0) + gate * delta_mu                    # [B, K, D]
        log_var = self.log_var_prior.unsqueeze(0) + gate * delta_logvar
        log_var = log_var.clamp(min=-10.0, max=2.0)
        weight_logits = self.weight_logits_prior.unsqueeze(0) + gate * delta_weight

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std                                               # [B, K, D]
        else:
            z = mu                                                           # deterministic at inference
        weights = F.softmax(weight_logits, dim=-1)                           # [B, K]
        return z, weights

    def kl_divergence(self):
        # KL( N(mu_prior, exp(log_var_prior)) || N(0, I) ) averaged over K*D
        return -0.5 * torch.mean(
            1 + self.log_var_prior - self.mu_prior.pow(2) - self.log_var_prior.exp()
        )


class GaussianCrossAttention(nn.Module):
    """
    Cross-attends an input sequence (or set of queries) against the input-conditioned
    Gaussian Latent History bank.  Applied twice in MYNET:
      1. On encoder output  [T, B, D]  — injects history into encoder memory.
      2. On decoder queries [K, B, D]  — injects history directly into anchor reps.

    The bank is shared across both call sites (same prototype dictionary, same
    adaptation heads) so parameters are not doubled.

    Per-head residual gates (one scalar per attention head, zero-init) replace the
    single global gate so each head can contribute a different amount.
    """
    def __init__(self, embedding_dim, num_heads, num_gaussians, dropout=0.1):
        super(GaussianCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.bank = GaussianHistoryBank(num_gaussians, embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        # Per-head gate: zero-init → no-op at init; each head learns independently
        self.residual_gate = nn.Parameter(torch.zeros(num_heads))

    def _apply_gate(self, attn_out):
        # attn_out: [S, B, D]  — expand per-head gate to full D via repeat
        D = attn_out.shape[-1]
        head_dim = D // self.num_heads
        # gate: [num_heads] → [1, 1, D] by repeating each scalar head_dim times
        gate = torch.sigmoid(self.residual_gate)             # [H]
        gate = gate.repeat_interleave(head_dim)              # [D]
        return attn_out * gate.view(1, 1, -1)

    def forward(self, x, bank_ctx=None):
        # x: [S, B, D] where S=T for encoder call, S=K for decoder call
        # bank_ctx: [T, B, D] encoder sequence used to build the bank context.
        #   When None, x itself is used (encoder call).  Pass encoded_x on decoder call
        #   so the bank always sees the full temporal context, not just K anchor tokens.
        ctx_src = x if bank_ctx is None else bank_ctx
        z, weights = self.bank.sample(ctx_src)                # z: [B, K, D], w: [B, K]
        memory = (z * weights.unsqueeze(-1)).permute(1, 0, 2) # [K, B, D]

        attn_out, _ = self.cross_attn(query=x, key=memory, value=memory)
        gated = self._apply_gate(self.dropout(attn_out))
        return self.norm(x + gated)

    def kl_loss(self):
        return self.bank.kl_divergence()


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 750, scale_factor: float = 1.0):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size * scale_factor)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class DualScaleTemporalEncoder(nn.Module):
    """
    Unified temporal encoder that captures both fine-grained local motion 
    and long-range temporal dependencies without relying on sequence-length branching.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(DualScaleTemporalEncoder, self).__init__()
        
        # Local scale: Depthwise 1D Convolutions for short-term temporal dynamics
        self.local_encoder = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            kernel_size=5, 
            padding=2, 
            groups=embedding_dim # Depthwise convolution for efficiency
        )
        self.local_norm = nn.LayerNorm(embedding_dim)
        
        # Global scale: Transformer Self-Attention for long-range dependencies
        self.global_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            activation='gelu',
            batch_first=False
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x shape: [seq_len, batch, dim]
        """
        seq_len, batch, dim = x.shape
        
        # Local processing
        local_x = x.permute(1, 2, 0)  # [batch, dim, seq_len]
        local_x = self.local_encoder(local_x)
        local_x = local_x.permute(2, 0, 1)  # [seq_len, batch, dim]
        local_x = self.local_norm(local_x + x)  # Residual connection
        
        # Global processing
        global_x = self.global_encoder(local_x)
        
        # Fusion
        fused = self.fusion(torch.cat([local_x, global_x], dim=-1))
        
        return fused

class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        self.anchors = opt["anchors"]
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        self.use_dse = bool(opt.get("DSE", False))
        self.use_glh = bool(opt.get("GLH", False))
        
        # Enhanced feature reduction with dynamic dropout
        self.feature_reduction_rgb = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2),
            nn.Dropout(dropout * 0.5)
        )
        self.feature_reduction_flow = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2),
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            n_embedding_dim, 
            dropout=dropout,
            maxlen=400,
            scale_factor=0.5
        )
        
        # Unified Dual-Scale Temporal Encoder (only built when --DSE is enabled)
        if self.use_dse:
            self.temporal_encoder = DualScaleTemporalEncoder(
                embedding_dim=n_embedding_dim,
                num_heads=n_enc_head,
                dropout=dropout
            )
        else:
            self.temporal_encoder = None
        
        # Main encoder (adaptive layers)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_enc_head, 
                dropout=dropout,
                activation='gelu'
            ) for i in range(n_enc_layer)
        ])
        self.encoder_norm = nn.LayerNorm(n_embedding_dim)
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_dec_head, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )
        

        
        # Enhanced classification and regression heads
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, 2)
        )

        self.decoder_token = nn.Parameter(torch.Tensor(len(self.anchors), 1, n_embedding_dim))
        nn.init.normal_(self.decoder_token, std=0.01)

        # GLH: one shared bank, applied at two points in the forward pass —
        # (1) on encoder memory to inject history context before decoding, and
        # (2) on decoder anchor queries directly so the heads see history-augmented reps.
        if self.use_glh:
            self.glh = GaussianCrossAttention(
                embedding_dim=n_embedding_dim,
                num_heads=n_enc_head,
                num_gaussians=opt.get("glh_gaussians", 8),
                dropout=dropout,
            )

        # Additional normalization layers
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Enhanced feature processing
        inputs = inputs.float()
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize
        seq_len = base_x.shape[0]
        
        # Apply positional encoding
        pe_x = self.positional_encoding(base_x)

        # Unified Dual-Scale Temporal Processing (skipped in baseline mode)
        if self.use_dse:
            encoded_x = self.temporal_encoder(pe_x)
        else:
            encoded_x = pe_x

        # Standard encoder processing
        for layer in self.encoder_layers:
            encoded_x = layer(encoded_x)
        
        # Apply encoder normalization
        encoded_x = self.encoder_norm(encoded_x)
        encoded_x = self.norm1(encoded_x)

        # GLH pass 1: augment encoder memory so the decoder cross-attends history-aware keys
        if self.use_glh:
            encoded_x = self.glh(encoded_x)

        # Decoder processing
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        decoded_x = self.decoder(decoder_token, encoded_x)

        # Add residual connection and normalization
        decoded_x = self.norm2(decoded_x + self.dropout1(decoder_token))

        # GLH pass 2: inject history directly into anchor queries using the full
        # encoder sequence as context (bank_ctx) so the bank always sees T tokens
        if self.use_glh:
            decoded_x = self.glh(decoded_x, bank_ctx=encoded_x)

        decoded_x = decoded_x.permute([1, 0, 2])

        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)

        return anc_cls, anc_reg


class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
