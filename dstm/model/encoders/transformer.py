import torch
from torch.nn import MultiheadAttention, LayerNorm, Linear, TransformerDecoderLayer, Dropout
from torch.nn.parameter import Parameter
from torch.nn.modules.transformer import Transformer, _get_activation_fn, _get_clones
import copy
import torch.nn.functional as F
import math
import pytorch_lightning as pl

def lin_proj(X, Wq, Wk, Wv):
        batch_size, seq_length, model_dim = X.shape
        num_heads, _ , hidden_dim = Wq.shape
        HX = torch.cat([ X for _ in range(num_heads)]).reshape(num_heads, batch_size,  seq_length, model_dim)  

        BWq = torch.cat([Wq for _ in range(batch_size)]).reshape(batch_size,  num_heads, model_dim, hidden_dim)
        HWqT = BWq.permute(1, 0, 3, 2) # num_heads, batch_size, elem_dim, elem_dim

        q_proj_flat = HX.reshape(-1, seq_length, model_dim).bmm(HWqT.reshape(-1, model_dim, hidden_dim)) # flattened batch mult
        q_proj = q_proj_flat.view(num_heads, batch_size, seq_length, hidden_dim)

        BWk = torch.cat([Wk for _ in range(batch_size)]).reshape(batch_size,  num_heads, model_dim, hidden_dim)
        HWkT = BWk.permute(1, 0, 3, 2) # num_heads, batch_size, elem_dim, seq_length

        k_proj_flat = HX.reshape(-1, seq_length, model_dim).bmm(HWkT.reshape(-1, model_dim, hidden_dim)) # flattened batch mult
        k_proj = k_proj_flat.view(num_heads, batch_size, seq_length, hidden_dim)

        

        BWv = torch.cat([Wv for _ in range(batch_size)]).reshape(batch_size,  num_heads, model_dim, hidden_dim)
        HWvT = BWv.permute(1, 0, 3, 2) # num_heads, batch_size, elem_dim, seq_length

        v_proj_flat = HX.reshape(-1, seq_length, model_dim).bmm(HWvT.reshape(-1, model_dim, hidden_dim)) # flattened batch mult
        b_proj = v_proj_flat.view(num_heads, batch_size, seq_length, hidden_dim)


        return q_proj, k_proj, b_proj
#TODO: should we use the non_flattened versions instead?
def att_logits(q_proj, k_proj): #, num_heads, batch_size):
    # ATT prior to softmax
    n_heads, batch_size, seq_length, elem_dim = q_proj.shape
    q_proj_flat = q_proj.view(-1, seq_length, elem_dim)
    k_proj_flat = k_proj.view(-1, seq_length, elem_dim)
    att_logits_flat = q_proj_flat.reshape(-1, seq_length, elem_dim).bmm(k_proj_flat.transpose(-1, -2))
    #att = attflat.reshape(num_heads, batch_size, seq_length, seq_length)
    att_logits = att_logits_flat.view(n_heads, batch_size, seq_length, seq_length)
    return att_logits

# # associciate elemts i,j with vector of size elem_dim # This is however only pr. head... according to Huang
# Rt = torch.rand(seq_length, elem_dim, seq_length)

# # For each projected sequenence element x_i -> z_i dot it by the relative interactions of the position a_{i,j}
# Sflat = resqflat.transpose(1,0).bmm(Rt).transpose(0, 1)
# S = Sflat.reshape(num_heads, batch_size, seq_length, seq_length)

# i = 3
# j = 2
# print(S[h, b, i, j ] - resq[h, b][i:i+1].mm(Rt[i,:,j].unsqueeze(1)))
# #print(S[h, b, i, j] - S[h, b, j, i])
# Relative with head (Shaw 2018)
def rel_pos_enc(q_proj, Rt): # Rt: num_heads, seq_length, elem_dim, seq_length
    num_heads = q_proj.shape[0]
    batch_size = q_proj.shape[1]
    seq_length = q_proj.shape[2]
    elem_dim = q_proj.shape[3]
    Rtflat = Rt.view(-1, elem_dim, seq_length)
    #resqflat = resq.view(-1, seq_length, elem_dim)
    Sflat = q_proj.transpose(1,2).reshape(-1, batch_size, elem_dim).bmm(Rtflat)
    #Sflat = resqflat.transpose(1,2).reshape(-1, batch_size, elem_dim).bmm(Rtflat)
    S = Sflat.view(num_heads, seq_length, batch_size, seq_length).transpose(1,2) # num_heads, batch_size, seq_length, seq_length
    return S

# Music Transformer (CZA Huang 2019)
def rel_pos_enc_eff(q_proj, ErT):
    num_heads, batch_size, seq_length, hidden_dim = q_proj.shape
    BErTflat = torch.cat([ErT for _ in range(batch_size)]) # batch_size*num_heads, elem_dim seq_length
    # TODO: might be inefficient
    HErT = BErTflat.reshape(batch_size, num_heads, hidden_dim, seq_length).transpose(0,1) #num_heads, batch_size, elem_dim seq_length
    HErTflat = HErT.reshape(-1, hidden_dim, seq_length)
    resqflat = q_proj.view(-1, seq_length, hidden_dim)
    relflat = resqflat.bmm(HErTflat)

    padded = torch.nn.functional.pad(relflat, (1, 0, 0, 0, 0, 0))
    reshaped = padded.reshape(-1, seq_length+1, seq_length)
    Sflat = reshaped[:, 1:, :]
    S = Sflat.view(num_heads, batch_size, seq_length, seq_length)
    return S


class MultiheadAttentionRelativeEncoding(torch.nn.Module):
    #TODO: now we have fixed legnths....
    def __init__(self, rel_clip_length, num_heads=7, model_dim=49, hidden_dim=None, dropout=0.0, device=None, dtype=None, pos_enc=True):
        super().__init__()
        factory_kwargs={'device': device, 'dtype': dtype}
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout = dropout
        self.rel_clip_length = rel_clip_length
        self.pos_enc = pos_enc
        if hidden_dim is None:
            if model_dim % num_heads:
                raise Exception("model_dim should be divisible with num_heads")
            hidden_dim = model_dim // num_heads
        self.hidden_dim = hidden_dim
        self.Wq = Parameter(torch.empty((self.num_heads, self.model_dim, self.hidden_dim), **factory_kwargs))
        self.Wk = Parameter(torch.empty((self.num_heads, self.model_dim, self.hidden_dim), **factory_kwargs))
        self.Wv = Parameter(torch.empty((self.num_heads, self.model_dim, self.hidden_dim), **factory_kwargs))
        self.Wo = Parameter(torch.empty((self.model_dim, self.model_dim), **factory_kwargs))
        if self.pos_enc:
            self.att_rel_emb = Parameter(torch.empty((self.num_heads, self.hidden_dim, self.rel_clip_length), **factory_kwargs)) # num_heads, elem_dim 
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Wq)
        torch.nn.init.xavier_uniform_(self.Wk)
        torch.nn.init.xavier_uniform_(self.Wv)
        torch.nn.init.xavier_uniform_(self.Wo)
        if self.pos_enc:
            torch.nn.init.xavier_uniform_(self.att_rel_emb)
        
    
    def calc_heads_parallel(self, X, mask, scale=True):
        q_proj, k_proj, v_proj = lin_proj(X, self.Wq, self.Wk, self.Wv)
        _, batch_size ,seq_length, _  = q_proj.shape 
        
        #scale
        if scale:
            q_proj /= math.sqrt(self.hidden_dim)
        #ordinary att
        logits = att_logits(q_proj, k_proj)
        #Srel = rel_pos_enc_eff(q_proj, self.att_rel_emb)
        if self.pos_enc:
            Srel = rel_pos_enc_eff(q_proj, self.att_rel_emb[:, :, (self.rel_clip_length-seq_length):])
            logits = logits + Srel + mask
        else:
            logits = logits + mask
        attn = torch.nn.functional.softmax(logits, dim=-1)
        if self.dropout > 0.0 and self.training:
            # TODO: change to module
            attn = torch.nn.functional.dropout(attn, p=self.dropout)
        #TODO add relative to values
        attn_flat = attn.view(-1, seq_length, seq_length)
        v_proj_flat = v_proj.view(-1, seq_length, self.hidden_dim)
        output_flat = attn_flat.bmm(v_proj_flat)
        output = output_flat.view(self.num_heads, batch_size, seq_length, self.hidden_dim)
        return output, attn
    
    def combine_heads(self, output):
        _, batch_size, seq_length, _ = output.shape
        output_perm = output.permute(1, 2, 0, 3)
        output_perm_flattened = output_perm.reshape(-1, self.model_dim)
        output_summed = output_perm_flattened.mm(self.Wo)
        return output_summed.view(batch_size, seq_length, self.model_dim) 

    def forward(self, X, mask):
        _, seq_length, _ = X.shape
        #mask = Transformer.generate_square_subsequent_mask(None, seq_length)
        output, _ = self.calc_heads_parallel(X, mask)
        return self.combine_heads(output)
        
        
        

class AbstractTransformerRNNLayer(torch.nn.Module):
    def __init__(self, num_heads, model_dim, hidden_dim, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        #super(TransformerDecoderLayer, self).__init__()
        super().__init__()
        self.model_dim = model_dim
        #NOTE: Calling like this gives a receptive field of d_model i.e. m
        self.self_attn = self.get_multi_head_attention(num_heads=num_heads, model_dim=model_dim, hidden_dim=hidden_dim, dropout=dropout,
                                            **factory_kwargs, **kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(model_dim, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, model_dim, **factory_kwargs)

        self.norm1 = LayerNorm(model_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(model_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
    
    def get_multi_head_attention(self, num_heads=7, model_dim=49, hidden_dim=None, dropout=0.0, device=None, dtype=None, **kwargs):
        raise NotImplementedError("Abstract method should be implimented")

    def forward_self_attention(self, tgt, tgt_mask, **kwargs):
        raise NotImplementedError("Abstract method should be implimented")
    def forward(self, tgt, tgt_mask = None, **kwargs):
        tgt2 = self.forward_self_attention(tgt, tgt_mask, **kwargs)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class RelativeEncodingTransformerRNNLayer(AbstractTransformerRNNLayer):
    #NOTE: adapted from  torch.nn.TransformerDecoderLayer
    def get_multi_head_attention(self, num_heads=7, model_dim=49, hidden_dim=None, dropout=0.0, device=None, dtype=None, rel_clip_length=512, pos_enc=True):
        return MultiheadAttentionRelativeEncoding(rel_clip_length=rel_clip_length, num_heads=num_heads, model_dim=model_dim, 
            hidden_dim=hidden_dim, dropout=dropout, device=device, dtype=dtype, pos_enc=pos_enc)

    def forward_self_attention(self, tgt, tgt_mask, **kwargs):
        return self.self_attn.forward(X=tgt, mask=tgt_mask,)

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        #self.scaling_embedding = torch.nn.Parameter(torch.rand(1)) #, **factory_kwargs)
        #torch.nn.init.xavier_uniform_(self.scaling_embedding)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = (torch.cos(position * div_term))[:, :d_model//2]
        self.register_buffer('pe', pe)

    def forward(self, x, scaling_embedding=0.1):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        #x = x + scaling_embedding*self.pe[:x.size(0)]
        x = x + x * scaling_embedding * self.pe[:x.size(0)]
        return x
        #return self.dropout(x)

class TransformerDecoder(torch.nn.Module):
  
    __constants__ = ['norm']

    def __init__(self, d, decoder_layer, num_layers=6, norm=None,abs_enc=False):
        super().__init__()
        
        self.layers = _get_clones(decoder_layer, num_layers)
        
        #decoder_layer, num_layers, d, m, norm=None)
        
        
        self.num_layers = num_layers
        self.norm = norm
        #self.m = m
        self.d = d
        if abs_enc:
            self.abs_enc = PositionalEncoding(d, decoder_layer.dropout.p, max_len=10000)
        else:
            self.abs_enc = False
        

    def forward(self, tgt):
        tgt = torch.nn.functional.pad(tgt, (self.layers[0].model_dim - self.d, 0, 1, 0, 0, 0))
        if self.abs_enc:
            tgt = self.abs_enc(tgt)
        #TODO: don't generate every time but use max length and then just index
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt.shape[1]).type_as(tgt) # .cuda( )
        output = tgt
        for mod in self.layers:
            output = mod(tgt=output, tgt_mask=tgt_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class StandardTransformerRNNLayer(TransformerDecoderLayer):
    #NOTE: adapted from  torch.nn.TransformerDecoderLayer
    __constants__ = ['batch_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        #NOTE: Calling like this gives a receptive field of d_model i.e. m
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask= None,
                tgt_key_padding_mask= None, memory_key_padding_mask= None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# def _get_clones(module, N):
#     return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# def _get_activation_fn(activation):
#     if activation == "relu":
#         return F.relu
#     elif activation == "gelu":
#         return F.gelu

#     raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

# def generate_square_subsequent_mask(sz):
#     #TODO: will be depreacted as should be static torch.nn.Transformer.generate_square_subsequent_mask, however, is not yet in pytorch "1.9"
#     return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1) 

class StandardTransformerRNN(torch.nn.Module):
    
    def __init__(self, m, d, activation='relu', dropout=0.0, nhead=7, num_layers=6):
        super().__init__()
        self.m = m
        self.d = d
        transformer_layer = StandardTransformerRNNLayer(d_model=m, nhead=nhead, dim_feedforward=2048, dropout=dropout, activation='relu', layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None)
        self.transformer_decoder = torch.nn.TransformerDecoder(transformer_layer, num_layers=num_layers)

    def forward(self, ss):
        tgt = torch.nn.functional.pad(ss, (self.m - self.d, 0, 1, 0, 0, 0))
        #TODO: memory have no effect but needs to be supplied when using TransformerDecoder. Alternatively write own TransformerDecoder
        memory = torch.zeros(ss.shape[0], 1, self.m) #.cuda()
        #TODO: don't generate every time but use max length and then just index
        mask = Transformer.generate_square_subsequent_mask(None, tgt.shape[1]).type_as(tgt) #.cuda()
        return self.transformer_decoder(tgt, memory, tgt_mask=mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
        #NOTE: since we need to project to a larger plane for now we just pad 

class LinearTransformer(torch.autograd.Function):
    """https://arxiv.org/pdf/2006.16236.pdf"""

    @staticmethod
    def forward(ctx, phi_Q, phi_K, V):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        num_heads, batch_size, seq_length, hidden_dim = phi_Q.shape 
        #_, batch_size ,seq_length, _  = q_proj.shape 
        #S = torch.zeros(num_heads, batch_size, hidden_dim, hidden_dim, dtype=phi_Q.dtype, device=phi_Q.device)
        #Z = torch.zeros(num_heads, batch_size, hidden_dim, dtype=phi_Q.dtype, device=phi_Q.device)
        #V_bar = torch.empty(num_heads, batch_size, seq_length, hidden_dim, dtype=phi_Q.dtype, device=phi_Q.device)
        # for i in range(seq_length):
        #     S = S + torch.matmul(phi_K[:, :, i, :, None], V[:,:,i, None, :])
        #     #Z = Z + phi_K[:,:,i, :]
        #     #output[:, :, i, :] = torch.matmul(phi_Q[:, :, i, None, :], S).squeeze(-2) / Z
        #     V_bar[:, :, i, :] = torch.matmul(phi_Q[:, :, i, None, :], S).squeeze(-2)
        S = torch.matmul(phi_K[:, :, :, :, None], V[:,:,:, None, :])
        S = S.cumsum(2)
        V_bar = torch.matmul(phi_Q[:, :, :, None, :], S).squeeze(-2)
        ctx.save_for_backward(phi_Q, phi_K, V, S)
        return V_bar
       
    @staticmethod
    def backward(ctx, G):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        #phi_Q, phi_K, V, = ctx.saved_tensors
        phi_Q, phi_K, V, S = ctx.saved_tensors
        #num_heads, batch_size, seq_length, hidden_dim = phi_Q.shape 
        #S = torch.zeros(num_heads, batch_size, hidden_dim, hidden_dim, dtype=phi_Q.dtype, device=phi_Q.device)
        #TODO: we could try to accumualte S, and store so we don't need to recompute...
        # grad_phi_Q = torch.empty(num_heads, batch_size, seq_length, hidden_dim, dtype=phi_Q.dtype, device=phi_Q.device)
        # for i in range(seq_length):
        #     S = S + torch.matmul(phi_K[:, :, i, :, None], V[:,:,i, None, :])
        #     grad_phi_Q[:, :, i, :] = torch.matmul(G[:, :, i, None, :], S.permute(0,1,3,2)).squeeze(-2)
        grad_phi_Q = torch.matmul(G[:, :, :, None, :], S.permute(0, 1, 2, 4, 3)).squeeze(-2)
        #S = torch.zeros(num_heads, batch_size, hidden_dim, hidden_dim, dtype=phi_Q.dtype, device=phi_Q.device)
        #grad_phi_K = torch.empty(num_heads, batch_size, seq_length, hidden_dim, dtype=phi_Q.dtype, device=phi_Q.device)
        #grad_V = torch.empty(num_heads, batch_size, seq_length, hidden_dim, dtype=phi_Q.dtype, device=phi_Q.device)
        # for i in range(seq_length-1, -1, -1):
        #     S = S + torch.matmul(phi_Q[:, :, i, :, None], G[:, :, i, None, :])
        #     #print(S.permute(0,1,3,2).shape)
        #     #print(phi_K[:, :, i, :, None].shape)
        #     grad_V[:, :, i, :] = torch.matmul(S.permute(0,1,3,2), phi_K[:, :, i, :, None]).squeeze(-1)
        #     grad_phi_K[:, :, i, :] = torch.matmul(S, V[:, :, i, :, None]).squeeze(-1)
        S = torch.matmul(phi_Q[:, :, :, :, None], G[:, :, :, None, :])
        # reverse cumsum https://github.com/pytorch/pytorch/issues/33520
        S = S +  torch.sum(S, dim=2, keepdims=True) - torch.cumsum(S, dim=2)
        grad_V = torch.matmul(S.permute(0, 1, 2, 4, 3), phi_K[:, :, :, :, None]).squeeze(-1)
        grad_phi_K = torch.matmul(S, V[:, :, :, :, None]).squeeze(-1)
        return grad_phi_Q, grad_phi_K, grad_V

linear_transformer = LinearTransformer.apply

class MultiheadAttentionLinear(torch.nn.Module):
    #TODO: now we have fixed legnths....
    def __init__(self, rel_clip_length, num_heads=7, model_dim=49, hidden_dim=None, dropout=0.0, device=None, dtype=None, pos_enc=True):
        super().__init__()
        factory_kwargs={'device': device, 'dtype': dtype}
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout = dropout
        self.rel_clip_length = rel_clip_length
        #self.pos_enc = pos_enc
        if hidden_dim is None:
            if model_dim % num_heads:
                raise Exception("model_dim should be divisible with num_heads")
            hidden_dim = model_dim // num_heads
        self.hidden_dim = hidden_dim
        self.Wq = Parameter(torch.empty((self.num_heads, self.model_dim, self.hidden_dim), **factory_kwargs))
        self.Wk = Parameter(torch.empty((self.num_heads, self.model_dim, self.hidden_dim), **factory_kwargs))
        self.Wv = Parameter(torch.empty((self.num_heads, self.model_dim, self.hidden_dim), **factory_kwargs))
        self.Wo = Parameter(torch.empty((self.model_dim, self.model_dim), **factory_kwargs))
        # if self.pos_enc:
        #     self.att_rel_emb = Parameter(torch.empty((self.num_heads, self.hidden_dim, self.rel_clip_length), **factory_kwargs)) # num_heads, elem_dim 
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Wq)
        torch.nn.init.xavier_uniform_(self.Wk)
        torch.nn.init.xavier_uniform_(self.Wv)
        torch.nn.init.xavier_uniform_(self.Wo)
        # if self.pos_enc:
        #     torch.nn.init.xavier_uniform_(self.att_rel_emb)
        
    
    def calc_heads_parallel(self, X, mask, scale=True):
        q_proj, k_proj, v_proj = lin_proj(X, self.Wq, self.Wk, self.Wv)
        q_proj = torch.nn.functional.elu(q_proj) + 1
        k_proj = torch.nn.functional.elu(k_proj) + 1
        #num_heads, batch_size, seq_length, hidden_dim = q_proj.shape 
        #_, batch_size ,seq_length, _  = q_proj.shape 
        # S = torch.zeros(num_heads, batch_size, hidden_dim, hidden_dim, dtype=X.dtype, device=X.device)
        # Z = torch.zeros(num_heads, batch_size, hidden_dim, dtype=X.dtype, device=X.device)
        # output = torch.empty(num_heads, batch_size, seq_length, hidden_dim, dtype=X.dtype, device=X.device)
        # for i in range(seq_length):
        #     S = S + torch.matmul(k_proj[:, :, i, :, None], v_proj[:,:,i, None, :])
        #     Z = Z + k_proj[:,:,i, :]
        #     output[:, :, i, :] = torch.matmul(q_proj[:, :, i, None, :], S).squeeze(-2) / Z
        # return output
        #TODO: dropout should actucally be inside linear_transformer function

        
        V_bar = linear_transformer(q_proj, k_proj, v_proj)
        Z = k_proj.cumsum(2)
        Z = torch.matmul(q_proj.unsqueeze(-2), Z.unsqueeze(-1)).squeeze(-1)
        return V_bar / Z
    #TODO: refactor to avoid code, dubl    
    def combine_heads(self, output):
        _, batch_size, seq_length, _ = output.shape
        output_perm = output.permute(1, 2, 0, 3)
        output_perm_flattened = output_perm.reshape(-1, self.model_dim)
        output_summed = output_perm_flattened.mm(self.Wo)
        return output_summed.view(batch_size, seq_length, self.model_dim) 

    def forward(self, X, mask):
        _, seq_length, _ = X.shape
        if self.training:
            #TODO: change to non functional
            X = torch.nn.functional.dropout(X, p=self.dropout)
        output = self.calc_heads_parallel(X, mask)
        output = self.combine_heads(output)
        
        return output

class LinearTransformerRNNLayer(AbstractTransformerRNNLayer):
    #NOTE: adapted from  torch.nn.TransformerDecoderLayer
    def get_multi_head_attention(self, num_heads=7, model_dim=49, hidden_dim=None, dropout=0.0, device=None, dtype=None, rel_clip_length=512, pos_enc=True):
        return MultiheadAttentionLinear(rel_clip_length=rel_clip_length, num_heads=num_heads, model_dim=model_dim, 
            hidden_dim=hidden_dim, dropout=dropout, device=device, dtype=dtype, pos_enc=pos_enc)

    def forward_self_attention(self, tgt, tgt_mask, **kwargs):
        return self.self_attn.forward(X=tgt, mask=tgt_mask,)
