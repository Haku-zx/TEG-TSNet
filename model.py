import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.distributions.normal import Normal

# global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Basic Blocks
# =========================================================

class Conv2d1x1(nn.Module):
    """1x1 Conv over (B, T, N, C) by using Conv2d on permuted layout."""
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu, bn_decay=None):
        super().__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]

        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x: (B, T, N, C)
        x = x.permute(0, 3, 2, 1)  # (B, C, N, T) -> treat as conv2d input
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1],
                       self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x.permute(0, 3, 2, 1)  # back to (B, T, N, C)


class MLP1x1(nn.Module):
    """Stacked 1x1 conv MLP for (B,T,N,C)."""
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super().__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert isinstance(units, list)

        self.layers = nn.ModuleList([
            Conv2d1x1(
                input_dims=input_dim,
                output_dims=num_unit,
                kernel_size=[1, 1],
                stride=[1, 1],
                padding='VALID',
                use_bias=use_bias,
                activation=activation,
                bn_decay=bn_decay
            )
            for input_dim, num_unit, activation in zip(input_dims, units, activations)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FeedForward(nn.Module):
    """Token-wise FFN over last dim (works on (B,T,N,D) directly)."""
    def __init__(self, fea, res_ln=False):
        super().__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x = self.ln(x + inputs)
        return x


# =========================================================
# 3.3 SPTD: Spatiotemporal Prior & Temporal Decomposition
# =========================================================

class TemporalEmbedding(nn.Module):
    """
    Build temporal embedding from (dayofweek, timeofday).
    Returns embeddings for history and future steps.
    """
    def __init__(self, input_dim, D, num_nodes, bn_decay):
        super().__init__()
        self.FC = MLP1x1(
            input_dims=[input_dim, D, D],
            units=[D, D, D],
            activations=[F.relu, F.relu, F.sigmoid],
            bn_decay=bn_decay
        )

    def forward(self, TE, SE, T_slots, num_vertex, num_his):
        # TE: (B, num_his+num_pred, 2)
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7, device=device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T_slots, device=device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for i in range(TE.shape[0]):
            timeofday[i] = F.one_hot(TE[..., 1][i].to(torch.int64) % T_slots, T_slots)

        X = torch.cat((timeofday, dayofweek), dim=-1)  # (B, T_all, 7+T_slots)
        X = X.unsqueeze(dim=2)  # (B, T_all, 1, 7+T_slots)
        X = X + torch.zeros(1, 1, num_vertex, 1, device=device)  # broadcast nodes
        X = self.FC(X)
        X = torch.sin(X)

        His = X[:, :num_his]
        Pred = X[:, num_his:]
        return His + F.relu(SE), Pred


class GraphSpectralEmbedding(nn.Module):
    """
    Graph Spectral Embedding (GSE) from Laplacian positional encoding (lpls).
    Your current lpls dim is 32 in utils.cal_lape().
    """
    def __init__(self, D, lape_dim=32):
        super().__init__()
        self.proj1 = nn.Linear(lape_dim, lape_dim)
        self.norm1 = nn.LayerNorm(lape_dim, elementwise_affine=False)
        self.act = nn.LeakyReLU()
        self.proj2 = nn.Linear(lape_dim, D)
        self.norm2 = nn.LayerNorm(D, elementwise_affine=False)

    def forward(self, lpls, batch_size, pred_steps):
        spa = self.norm2(self.proj2(self.act(self.norm1(self.proj1(lpls)))))
        spa = spa.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).repeat(1, pred_steps, 1, 1)
        return torch.sigmoid(spa)


class TrendSeasonalDecomposition(nn.Module):
    """
    Learnable Trend-Seasonal Decomposition (TSD):
    trend = X ⊙ STEmb, seasonal = X - trend
    """
    def __init__(self):
        super().__init__()

    def forward(self, X, STEmb_his):
        trend = torch.mul(X, STEmb_his)
        seasonal = X - trend
        return trend, seasonal


# =========================================================
# 3.4 TEG-Encoder: Tensor-Evolving Graph + Diffusion Graph Encoding
# =========================================================

class NConv(nn.Module):
    def forward(self, x, A):
        # x: (B, C, T, N) or after transpose: see DiffusionGraphEncoder
        return torch.einsum('ncvl,nwv->ncwl', (x, A)).contiguous()


class DiffusionGraphEncoder(nn.Module):
    """
    Diffusion graph convolution encoder using dynamic adjacency.
    x: (B, T, N, D)
    support: list of adjacency tensors, each (B, N, N)
    """
    def __init__(self, c_in, c_out, dropout=0.3, support_len=1, order=2, bn_decay=0.1):
        super().__init__()
        self.nconv = NConv()
        self.order = order
        self.dropout = dropout

        c_in_eff = (order * support_len + 1) * c_in
        self.mlp = MLP1x1(c_in_eff, c_out, activations=F.relu, bn_decay=bn_decay)

    def forward(self, x, support):
        # x: (B, T, N, D) -> (B, D, T, N)
        x = x.transpose(1, 3)
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)   # (B, D*(...), T, N)
        h = h.transpose(1, 3)       # (B, N, T, C) -> via MLP1x1 expects (B,T,N,C)
        h = self.mlp(h)
        h = h.transpose(1, 3)
        h = F.dropout(h, self.dropout, training=self.training)
        h = h.transpose(1, 3)       # back to (B,T,N,D)
        return h


class GRUCellLite(nn.Module):
    def __init__(self, outfea):
        super().__init__()
        self.ff = nn.Linear(2 * outfea, 2 * outfea)
        self.zff = nn.Linear(2 * outfea, outfea)
        self.outfea = outfea

    def forward(self, x, h):
        r, u = torch.split(torch.sigmoid(self.ff(torch.cat([x, h], -1))), self.outfea, -1)
        z = torch.tanh(self.zff(torch.cat([x, r * h], -1)))
        h = u * z + (1 - u) * h
        return h


class GRUEncoder(nn.Module):
    def __init__(self, outfea, num_step):
        super().__init__()
        self.cells = nn.ModuleList([GRUCellLite(outfea) for _ in range(num_step)])

    def forward(self, x):
        B, T, N, Fdim = x.shape
        h = torch.zeros([B, N, Fdim], device=device)
        outs = []
        for t in range(T):
            h = self.cells[t](x[:, t, :, :], h)
            outs.append(h)
        return torch.stack(outs, 1)  # (B, T, N, D)


# =========================================================
# 3.5 SMoE-Decoder: Spatiotemporal Attention Fusion + Sparse MoE
# =========================================================

class MAB(nn.Module):
    def __init__(self, K, d, input_dim, output_dim, bn_decay):
        super().__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = MLP1x1(input_dim, D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = MLP1x1(input_dim, D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = MLP1x1(input_dim, D, activations=F.relu, bn_decay=bn_decay)
        self.FC   = MLP1x1(D, output_dim, activations=F.relu, bn_decay=bn_decay)

        self.last_attention = None

    def forward(self, Q, K_in, batch_size, attn_type="spatial", mask=None):
        query = self.FC_q(Q)
        key   = self.FC_k(K_in)
        value = self.FC_v(K_in)

        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key   = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        if attn_type == "temporal":
            query = query.permute(0, 2, 1, 3)
            key   = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            if mask is not None and mask.shape == query.shape:
                mask = mask.permute(0, 2, 1, 3)

        attn = torch.matmul(query, key.transpose(2, 3)) / (self.d ** 0.5)
        if mask is not None:
            # build compatibility mask
            if mask.shape == query.shape:
                set_mask = torch.ones_like(key, device=device)
                mask2 = torch.matmul(mask, set_mask.transpose(2, 3))
            elif mask.shape == key.shape:
                set_mask = torch.ones_like(query, device=device)
                mask2 = torch.matmul(set_mask, mask.transpose(2, 3))
            else:
                mask2 = None
            if mask2 is not None:
                attn = attn.masked_fill(mask2 == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        self.last_attention = attn

        out = torch.matmul(attn, value)
        if attn_type == "temporal":
            out = out.permute(0, 2, 1, 3)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.FC(out)
        return out


class TemporalSeparationAttention(nn.Module):
    """(Optional) kept for compatibility if you later use temporal attention mask."""
    def __init__(self, K, d, num_of_vertices, set_dim, bn_decay):
        super().__init__()
        D = K * d
        self.I = nn.Parameter(torch.Tensor(1, set_dim, num_of_vertices, D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(K, d, D, D, bn_decay)
        self.mab1 = MAB(K, d, D, D, bn_decay)

    def forward(self, X, mask):
        B = X.shape[0]
        I = self.I.repeat(B, 1, 1, 1)
        H = self.mab0(I, X, B, "temporal", mask)
        out = self.mab1(X, H, B, "temporal", mask)
        return X + out


class SpatiotemporalAttentionFusion(nn.Module):
    """
    SAF: decoder attention fusion over (X, TE_future, SE).
    """
    def __init__(self, K, d, num_of_vertices, set_dim, bn_decay):
        super().__init__()
        D = K * d
        self.I = nn.Parameter(torch.Tensor(1, set_dim, num_of_vertices, 3 * D))
        nn.init.xavier_uniform_(self.I)


    def forward(self, X, TE_pred, SE, mask):
        B = X.shape[0]

        return X_mid + out


# ---------------- Sparse MoE ----------------

class ExpertMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
        )
        return zeros.index_add(0, self._batch_index, stitched.float())


class SparseGatedMoE(nn.Module):
    def __init__(self, model_dim, num_experts=20, hidden_dim=None, k=2, noisy_gating=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * model_dim
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating

        self.experts = nn.ModuleList([ExpertMLP(model_dim, hidden_dim, model_dim) for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(model_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(model_dim, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self._normal = Normal(self.mean, self.std)

        assert k <= num_experts

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        prob_if_in = self._normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self._normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits



        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x):
        # x: [B*, D]
        gates, _load = self.noisy_top_k_gating(x, self.training)

        y = dispatcher.combine(expert_outputs)
        return y


class SMoEFFN(nn.Module):
    """Apply SparseGatedMoE token-wise to (..., D)."""
    def __init__(self, model_dim, num_experts=20, hidden_dim=None, k=2):
        super().__init__()
        self.model_dim = model_dim
        self.moe = SparseGatedMoE(model_dim, num_experts=num_experts, hidden_dim=hidden_dim, k=k)

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.model_dim)
        y_flat = self.moe(x_flat)
        return y_flat.view(*orig_shape)


# =========================================================
# Main Model: TEG-TSNet
# =========================================================

class TEGTSNet(nn.Module):
    """
    TEG-TSNet: Tensor-Evolving Graph with Temporal Separation Network.
    """
    def __init__(self, args, bn_decay):
        super().__init__()
        data_config = args['Data']
        training_config = args['Training']

        L = int(training_config['L'])
        K = int(training_config['K'])
        d = int(training_config['d'])
        self.L = L
        self.K = K
        self.d = d

        D = K * d
        self.num_his = int(training_config['num_his'])
        self.num_pred = int(training_config['num_pred'])

        time_slice_size = int(data_config['time_slice_size'])
        self.T_slots = int(1440 / time_slice_size)  # slots per day
        self.time_input_dim = self.T_slots + 7

        self.num_of_vertices = int(data_config['num_of_vertices'])
        set_dim = int(training_config['reference'])
        self.order = int(training_config['order'])

        out_channels = int(training_config['out_channels'])

        # SPTD
        self.spa_gse = GraphSpectralEmbedding(D=D, lape_dim=32)
        self.tem_emb = TemporalEmbedding(self.time_input_dim, D, self.num_of_vertices, bn_decay)
        self.tsd = TrendSeasonalDecomposition()
        self.ffn_trend = FeedForward([D, D], res_ln=True)
        self.ffn_seasonal = FeedForward([D, D], res_ln=True)

        # TEG-Encoder
        self.trend_encoder = GRUEncoder(D, self.num_his)
        self.seasonal_encoder = GRUEncoder(D, self.num_his)

        # Tensor-based graph generation parameters (TB2G)
        self.nodevec_time = nn.Parameter(torch.randn(self.T_slots, D).to(device), requires_grad=True)
        self.nodevec_src  = nn.Parameter(torch.randn(self.num_of_vertices, D).to(device), requires_grad=True)
        self.nodevec_tgt  = nn.Parameter(torch.randn(self.num_of_vertices, D).to(device), requires_grad=True)
        self.core_tensor  = nn.Parameter(torch.randn(D, D, D).to(device), requires_grad=True)

        self.dge = DiffusionGraphEncoder(D, D, order=self.order)

        # Decoder: SAF + SMoE
        self.decoder_blocks = nn.ModuleList([
            SpatiotemporalAttentionFusion(K, d, self.num_of_vertices, set_dim, bn_decay)
            for _ in range(L)
        ])
        self.smoe_ffn = SMoEFFN(model_dim=D, num_experts=20, hidden_dim=4 * D, k=2)
        self.dec_norm = nn.LayerNorm(D, elementwise_affine=False)

        # IO projections
        self.in_proj = MLP1x1(input_dims=[1, D], units=[D, D], activations=[F.relu, None], bn_decay=bn_decay)
        self.out_proj = MLP1x1(input_dims=[D, D], units=[D, out_channels], activations=[F.relu, None], bn_decay=bn_decay)

    def tb2g_generate_adjacency(self, time_embedding, src_embedding, tgt_embedding, core_tensor):
        """
        TB2G: Tensor-Based Graph Generation:
        A_dyn = softmax(relu( einsum(time, core) * src * tgt ), dim=target )
        output: (B, N, N) row-normalized.
        """
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_tensor)
        adp = torch.einsum('bj, ajk->abk', src_embedding, adp)
        adp = torch.einsum('ck, abk->abc', tgt_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, X, TE, lpls, mode, type='train', return_intermediate=False):
        """
        X: (B, T_his, N, 1)
        TE: (B, T_his+T_pred, 2)
        lpls: (N, 32)
        """
        X = self.in_proj(X)  # (B, T_his, N, D)
        fc1_out = X if return_intermediate else None

        # dynamic adjacency based on current time slot index (use TE[:,0,1])
        ind = TE[:, 0, 1]
        ind = torch.tensor(ind, dtype=torch.long, device=device)
        A_dyn = self.tb2g_generate_adjacency(self.nodevec_time[ind], self.nodevec_src, self.nodevec_tgt, self.core_tensor)
        supports = [A_dyn]

        # DGE encoding (inject A_dyn)
        X = self.dge(X, supports)
        gcn_out = X if return_intermediate else None

        # spatial prior (GSE) expanded on pred steps
        SE = self.spa_gse(lpls, X.shape[0], self.num_pred)  # (B, T_pred, N, D)

        # temporal embedding for his/pred
        his_te, pred_te = self.tem_emb(TE, SE, self.T_slots, self.num_of_vertices, self.num_his)

        # TSD (trend/seasonal)
        trend, seasonal = self.tsd(X, his_te)
        trend = self.ffn_trend(trend)
        seasonal = self.ffn_seasonal(seasonal)

        # two-branch GRU encoders
        trend_h = self.trend_encoder(trend)
        seasonal_h = self.seasonal_encoder(seasonal)

        trend_enc = trend_h if return_intermediate else None
        seasonal_enc = seasonal_h if return_intermediate else None

        H = trend_h + seasonal_h  # SAF input

        decoder_outputs = []
        for blk in self.decoder_blocks:
            H = blk(H, pred_te, SE, None)     # SAF
            H = self.dec_norm(H + self.smoe_ffn(H))  # SMoE FFN
            if return_intermediate:
                decoder_outputs.append(H)

        Y = self.out_proj(H)  # (B, T_his, N, out_channels)  (same T as H)

        if return_intermediate:
            intermediates = {
                "fc1_out": fc1_out,
                "A_dyn": A_dyn,
                "dge_out": gcn_out,
                "trend_enc": trend_enc,
                "seasonal_enc": seasonal_enc,
                "final_hidden": H,
                "decoder_outputs": decoder_outputs,
            }
            return Y, intermediates
        return Y


def build_model(config, bn_decay=0.1):
    model = TEGTSNet(config, bn_decay=bn_decay).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
