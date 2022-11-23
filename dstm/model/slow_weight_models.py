import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class DilatedConvBlock(torch.nn.Module):
    def __init__(self, m, d, activation="relu", filters=[(128, 128, 128, 128)], dropout=0.0, residual=True):
        """[summary]

        Args:
            m (int): number of discrete codes
            d (int): dimension of output
            filters list(tuple): filters in first layer. List of tuples of number of filter and receptive field

        """
        super().__init__()
        self.activation = activation
        self.residual =residual
        cnns = []
        #self.r = 2
        self.dropout = torch.nn.Dropout(p=dropout)
        if activation == "gated_activation":
        # Gated activation
            cnns.append(torch.nn.Conv2d(
                        1, 2*filters[0], kernel_size=(2, d)))
            for i in range(1, len(filters)):
                cnns.append(torch.nn.Conv2d(
                    filters[i-1], 2*filters[i], dilation=(2**i, 1), kernel_size=(2, 1)))
            cnns.append(torch.nn.Conv2d(
                    filters[-1], 2*m, dilation=(2**(len(filters)), 1), kernel_size=(2, 1)))
        else:
            cnns.append(torch.nn.Conv2d(
                        1, filters[0], kernel_size=(2, d)))
            for i in range(1, len(filters)):
            #TODO: readded bug
            #for i in range(1, len(filters) - 1):
                cnns.append(torch.nn.Conv2d(
                    filters[i-1], filters[i], dilation=(2**i, 1), kernel_size=(2, 1)))
            cnns.append(torch.nn.Conv2d(
                    filters[-1], m, dilation=(2**(len(filters)), 1), kernel_size=(2, 1)))

        self.cnns = torch.nn.ModuleList(cnns)

    def forward(self, ss):
        """[summary]

        Args:
            ss (Tensor): BxTxd sequence of T previous tokens

        Returns:
            probs (Tensor): Bx(T+1)xd sequence of probs/one-hot
            logits (Tensor): Bx(T+1)xd sequence of logits
        """
        in_ = ss.unsqueeze(1)
        if self.activation == "gated_activation":
            #Gated activation
            for i, cnn in enumerate(self.cnns):
                residual = in_
                if i < len(self.cnns) -1:
                    in_ = F.pad(in_, (0, 0, cnn.dilation[0], 0, 0, 0))
                else:
                    in_ = F.pad(in_, (0, 0, self.cnns[-1].dilation[0] + 1, 0, 0, 0))
                in_ = self.dropout(in_)
                in_ = cnn(in_)
                n_filters = in_.shape[1] // 2
                ouput = torch.tanh(in_[:, :n_filters, :, :])
                gates = torch.sigmoid(in_[:, n_filters:, :, :])
                in_ = ouput * gates
                if i > 0 and i < len(self.cnns) -1 and self.residual:
                    in_ += residual
        else:
            if self.activation == "relu":
                activation_fn = torch.nn.ReLU()
            elif self.activation == "selu":
                activation_fn = torch.nn.SELU()
            else:
                activation_fn = torch.nn.Identity()
            for i, cnn in enumerate(self.cnns[:-1]):
                residual = in_
                in_ = F.pad(in_, (0, 0, cnn.dilation[0], 0, 0, 0))
                in_ = self.dropout(in_)
                in_ = activation_fn(cnn(in_))
                if i > 0 and self.residual:
                    in_ = in_ + residual
            in_ = F.pad(in_, (0, 0, self.cnns[-1].dilation[0] + 1, 0, 0, 0))
            in_ = self.dropout(in_)
            in_ = self.cnns[-1](in_)
        logits = in_.squeeze(-1)
        logits = logits.permute(0, 2, 1)
        return logits