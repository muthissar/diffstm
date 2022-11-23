import torch
from dstm.model.slow_weight_models import DilatedConvBlock
from dstm.model.encoders.transformer import StandardTransformerRNN, TransformerDecoder, RelativeEncodingTransformerRNNLayer, LinearTransformerRNNLayer
from dstm.model.module import Module
class LTM(Module):
    def __init__(self, d, hparams):
        """
        Args:
            d ([type]): [description]
            hparams (obj): namespace of hyper parameters
        """
        super().__init__(d, hparams)
        if hparams.encoder == "transformer_abs":
            raise NotImplementedError("Need absolute encoding")
            self.slow_weight_model = StandardTransformerRNN(m=d, d=d, dropout=hparams.dropout, num_layers=hparams.transformer_layers, nhead=hparams.transformer_n_head)
        elif hparams.encoder == "transformer_rel":
            #TODO:
            decoder_layer = RelativeEncodingTransformerRNNLayer(num_heads=hparams.transformer_n_head, model_dim=d, hidden_dim=None, 
                dropout=hparams.dropout, device=None, dtype=None, rel_clip_length=hparams.seq_max_length+1, pos_enc=True)
            self.slow_weight_model = TransformerDecoder(d=d, decoder_layer=decoder_layer, num_layers=hparams.transformer_layers, abs_enc=False)
        elif hparams.encoder == "transformer_lin":
            decoder_layer = LinearTransformerRNNLayer(num_heads=hparams.transformer_n_head, model_dim=d, hidden_dim=None, 
                dropout=hparams.dropout, device=None, dtype=None, rel_clip_length=hparams.seq_max_length+1, pos_enc=False)
            self.slow_weight_model = TransformerDecoder(d=d, decoder_layer=decoder_layer, num_layers=hparams.transformer_layers, abs_enc=True)
        
        else:
            self.slow_weight_model = DilatedConvBlock(m=d, d=d, filters=hparams.filters, activation=hparams.activation, dropout=hparams.dropout)
    
    @staticmethod
    def add_model_specific_args(parser):
        pass
    def forward(self, h_src, s_tar, W):
        raise NotImplementedError("forward")
        #return pred, W

    def probs(self, batch, hard=False):
        """Returns the probabilities for full sequence

        Args:
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        logits = self.slow_weight_model(batch)
        probs = torch.nn.functional.softmax(logits, dim=2)[:, :-1, :]
        return probs, -1