import torch
from torch import nn, einsum
from einops import rearrange, repeat

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


List = nn.ModuleList


# normalizations
class PreNorm(nn.Module):
    def __init__(
        self,
        dim,
        fn
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args,**kwargs)


# gated residual
class Residual(nn.Module):
    def forward(self, x, res):
        return x + res


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, 1, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim = -1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


# attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        pos_emb = None,
        dim_head = 64,
        heads = 8,
        edge_dim = None
    ):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges=None, mask=None, edge_mask=None, cross_kv=None):
        h = self.heads

        q = self.to_q(nodes)
        if cross_kv is None:
            k, v = self.to_kv(nodes).chunk(2, dim = -1)
        else:
            k, v = self.to_kv(cross_kv).chunk(2, dim = -1)

        if edges is not None:
            e_kv = self.edges_to_kv(edges)
            e_kv = rearrange(e_kv, 'b ... (h d) -> (b h) ... d', h = h)

        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> (b h) ... d', h = h), (q, k, v))

        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(nodes.shape[1], device = nodes.device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        if exists(edge_mask):
            pass
        
        if edges is not None:
            k, v = map(lambda t: rearrange(t, 'b j d -> b () j d '), (k, v))
            ek, ev = e_kv, e_kv
            k = k + ek
            v = v + ev

        if edges is not None:
            sim = einsum('b i d, b i j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            # mask = rearrange(mask, 'b i -> b i ()') & rearrange(mask, 'b j -> b () j')
            if edges is not None:
                mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            else:
                mask = rearrange(mask, 'b d -> b () d')
                mask = repeat(mask, 'b i j -> (b h) i j', h=h)   # 为什么要repeat？
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        if edges is not None:
            out = einsum('b i j, b i j d -> b i d', attn, v)
        else:
            out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# optional feedforward
def FeedForward(dim, ff_mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * ff_mult),
        nn.GELU(),
        nn.Linear(dim * ff_mult, dim)
    )


# classes
class GraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        h_dim,
        depth,
        attention,
        dim_head=64,
        edge_input_dim=None,
        edge_h_dim=None,
        heads=8,
        gated_residual=True,
        with_feedforwards = False,
        norm_edges = False,
        rel_pos_emb = False,
    ):
        super().__init__()
        self.layers = List([])
        edge_h_dim = default(edge_h_dim, h_dim)

        self.n_emb = nn.Linear(input_dim, h_dim)
        self.e_emb = nn.Linear(edge_input_dim, edge_h_dim)
        self.norm_edges = nn.LayerNorm(edge_h_dim) if norm_edges else nn.Identity()

        # assert h_dim % heads == 0
        # dim_head = h_dim // heads

        # pos_emb = RotaryEmbedding(dim_head) if rel_pos_emb else None

        for i in range(depth):
            self.layers.append(List([
                List([
                    # PreNorm(h_dim, Attention(h_dim, pos_emb=pos_emb, edge_dim=edge_h_dim, dim_head=dim_head, heads=heads)),
                    PreNorm(h_dim, attention[i]),
                    GatedResidual(h_dim)
                ]),
                List([
                    PreNorm(h_dim, FeedForward(h_dim)),
                    GatedResidual(h_dim)
                ]) if with_feedforwards else None
            ]))

    def forward(self, nodes, edges=None, lengths=None, adj=None):
        nodes = self.n_emb(nodes)
        if edges is not None:
            edges = self.e_emb(edges)
            edges = self.norm_edges(edges)

        mask = None
        if isinstance(adj, torch.Tensor):
            mask = ~sequence_mask(lengths).unsqueeze(1) + ~adj.bool()
            # mask = ~sequence_mask(lengths).unsqueeze(1)
        elif adj is None:
            mask = ~sequence_mask(lengths).unsqueeze(1)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, edges, mask=mask), nodes)

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        return nodes, edges


class TextTransformer(nn.Module):
    def __init__(
        self,
        h_dim,
        depth,
        attention,
        with_feedforwards = False,
    ):
        super().__init__()
        self.layers = List([])

        for i in range(depth):
            self.layers.append(List([
                List([
                    PreNorm(h_dim, attention[i]),
                    GatedResidual(h_dim)
                ]),
                List([
                    PreNorm(h_dim, FeedForward(h_dim)),
                    GatedResidual(h_dim)
                ]) if with_feedforwards else None
            ]))

    def forward(self, text, node=None, mask = None):
        
        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            text = attn_residual(attn(text, edges=None, mask=mask, cross_kv=node), text)

            if exists(ff_block):
                ff, ff_residual = ff_block
                text = ff_residual(ff(text), text)

        return text
    


if __name__ == '__main__':
    model = GraphTransformer(
        input_dim=256,
        h_dim=256,
        depth=6,
        edge_h_dim=512,  # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
        with_feedforwards=True,
        # whether to add a feedforward after each attention layer, suggested by literature to be needed
        gated_residual=True,  # to use the gated residual to prevent over-smoothing
        rel_pos_emb=True  # set to True if the nodes are ordered, default to False
    )

    nodes = torch.randn(1, 128, 256)
    edges = torch.randn(1, 128, 128, 512)
    c = torch.zeros(512)
    edges[0][0][0] = c
    mask = torch.ones(1, 128).bool()
    mask[0][0] = False

    nodes, edges = model(nodes, edges, lengths=2)

    print(nodes.shape)  # (1, 128, 256) - project to R^3 for coordinates

