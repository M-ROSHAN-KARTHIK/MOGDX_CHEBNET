import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

orig_sys_path = sys.path[:]
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dirname)
from GNN_MME import Encoder
sys.path.insert(0, os.path.join(dirname, '../Modules/PNetTorch/MAIN'))
from Pnet import PNET
sys.path = orig_sys_path


class SparseChebConv(nn.Module):
    """Chebyshev polynomial graph convolution implemented with sparse matrix multiplications.
    This is a practical ChebNet-style layer for full-graph training.
    """
    def __init__(self, in_feats, out_feats, cheb_k=2, bias=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.cheb_k = int(cheb_k)
        self.linears = nn.ModuleList([
            nn.Linear(in_feats, out_feats, bias=bias) for _ in range(self.cheb_k + 1)
        ])

    def forward(self, x, adj_norm):
        tx_0 = x
        out = self.linears[0](tx_0)
        if self.cheb_k >= 1:
            tx_1 = torch.sparse.mm(adj_norm, x)
            out = out + self.linears[1](tx_1)
        else:
            return out

        for k in range(2, self.cheb_k + 1):
            tx_2 = 2.0 * torch.sparse.mm(adj_norm, tx_1) - tx_0
            out = out + self.linears[k](tx_2)
            tx_0, tx_1 = tx_1, tx_2
        return out


class ChebNetII_MME(nn.Module):
    """Multi-modal encoder + Chebyshev spectral GNN for patient similarity graphs.

    Uses the same encoder idea as the original project, but replaces GraphConv with
    full-graph Chebyshev polynomial aggregation so wider neighborhoods can be modeled
    more flexibly than a traditional first-order GCN.
    """
    def __init__(self, input_dims, encoder_dims, latent_dims, decoder_dim, hidden_feats,
                 num_classes, dropout=0.5, enc_dropout=0.5, cheb_k=2, PNet=None):
        super().__init__()
        self.encoder_dims = nn.ModuleList()
        self.gnnlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.input_dims = input_dims
        self.hidden_feats = hidden_feats
        self.num_classes = num_classes
        self.cheb_k = int(cheb_k)
        self.uses_full_graph = True
        self._cached_adj = None
        self._cached_num_nodes = None

        for modality in range(len(input_dims)):
            if PNet is not None:
                self.encoder_dims.append(
                    PNET(
                        reactome_network=PNet,
                        input_dim=input_dims[modality],
                        output_dim=decoder_dim,
                        activation=nn.ReLU,
                        dropout=enc_dropout,
                        filter_pathways=True,
                        input_layer_mask=None,
                    )
                )
            else:
                self.encoder_dims.append(
                    Encoder(
                        input_dims[modality],
                        encoder_dims[modality],
                        latent_dims[modality],
                        decoder_dim,
                        dropout=enc_dropout,
                    )
                )

        prev_dim = decoder_dim
        for hidden_dim in hidden_feats:
            self.gnnlayers.append(SparseChebConv(prev_dim, hidden_dim, cheb_k=self.cheb_k))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        self.out_layer = SparseChebConv(prev_dim, num_classes, cheb_k=self.cheb_k)
        self.drop = nn.Dropout(dropout)

    def _encode_modalities(self, h):
        prev_dim = 0
        encoded_modalities = []

        for encoder, dim in zip(self.encoder_dims, self.input_dims):
            x_mod = h[:, prev_dim:prev_dim + dim]
            n = x_mod.shape[0]
            nan_rows = torch.isnan(x_mod).any(dim=1)
            valid_x = x_mod[~nan_rows]

            if valid_x.shape[0] == 0:
                # fallback if an entire modality is missing in a batch
                encoded = torch.zeros((n, encoder.decoder[0].out_features if hasattr(encoder, 'decoder') else 32),
                                      device=h.device, dtype=h.dtype)
            else:
                enc_valid = encoder(valid_x)
                fill_value = torch.median(enc_valid, dim=0).values
                encoded = fill_value.unsqueeze(0).repeat(n, 1)
                encoded[~nan_rows] = enc_valid
            encoded_modalities.append(encoded)
            prev_dim += dim

        x = torch.stack(encoded_modalities, dim=0)
        x = torch.mean(x, dim=0)
        return x

    def _build_normalized_adj(self, g, device):
        num_nodes = g.num_nodes()
        if self._cached_adj is not None and self._cached_num_nodes == num_nodes and self._cached_adj.device == device:
            return self._cached_adj

        src, dst = g.edges()
        src = src.to(device)
        dst = dst.to(device)
        nodes = torch.arange(num_nodes, device=device)

        src_all = torch.cat([src, dst, nodes])
        dst_all = torch.cat([dst, src, nodes])
        values = torch.ones(src_all.shape[0], device=device)

        adj = torch.sparse_coo_tensor(
            torch.stack([src_all, dst_all]), values, (num_nodes, num_nodes), device=device
        ).coalesce()

        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        row, col = adj.indices()
        norm_vals = adj.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(
            adj.indices(), norm_vals, adj.shape, device=device
        ).coalesce()

        self._cached_adj = adj_norm
        self._cached_num_nodes = num_nodes
        return adj_norm

    def forward(self, h, g):
        x = self._encode_modalities(h)
        adj_norm = self._build_normalized_adj(g, x.device)

        for layer, bn in zip(self.gnnlayers, self.batch_norms):
            x = layer(x, adj_norm)
            x = bn(x)
            x = F.relu(x)
            x = self.drop(x)
        x = self.out_layer(x, adj_norm)
        return x

    def embedding_extraction(self, g, h, device=None, batch_size=None):
        x = self._encode_modalities(h)
        adj_norm = self._build_normalized_adj(g, x.device)
        for layer, bn in zip(self.gnnlayers, self.batch_norms):
            x = layer(x, adj_norm)
            x = bn(x)
            x = F.relu(x)
        return x

    def feature_importance(self, *args, **kwargs):
        raise NotImplementedError('Feature importance for ChebNetII_MME is not implemented in this student update.')

    def layerwise_importance(self, *args, **kwargs):
        raise NotImplementedError('Layerwise importance for ChebNetII_MME is not implemented in this student update.')
