import torch
import torch.nn.functional as F
from dstm.model.slow_weight_models import DilatedConvBlock
import numpy as np
from dstm.model.coded_short_term_model import *
from dstm.model.module import Module
import argparse

def restricted_float(min_, max_):
    def restricted_float_(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("{} not a floating-point literal".format(x,))
        if x <= min_ or x > max_:
            raise argparse.ArgumentTypeError("{} not in range ({}, {}]".format(x, min_, max_))
        return x
    return restricted_float_

class DSTM(Module):
    def __init__(self, d, hparams):
        """
        Args:
            d ([type]): [description]
            hparams (obj): namespace of hyper parameters
        """
        super().__init__(d, hparams)
        self.m = hparams.encoder_output_dim
        self.W_ = None
        self.probs_ = None
        if hparams.generate_target_pitch:
            m = self.m + d
            filters = np.array(hparams.filters) + d
        else:
            m = self.m
            filters = np.array(hparams.filters)
        self.slow_weight_model = DilatedConvBlock(m=m, d=d, filters=filters, activation=hparams.activation, dropout=hparams.dropout)
    def _create_lambda_with_globals(s):
        return eval(s, globals())
    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--gumbel_temperature', type=restricted_float(0,float('inf')), default=1, help="temperature controlling the approximation of gumbel sampels to one-hot encoded categorical samples")
        parser.add_argument('--gumbel_hard', action='store_true', default=False, help="straight through gumbel softmax")
        parser.add_argument('--simulated_annealing', type=str, default='lambda gradient_step: max(0.8, -.7/90000*gradient_step + 1.5)', help="lambda function string representation of gumbel temperature as a function of number of gradient steps.")
        parser.add_argument('--short_term_method', type=str, choices=["gumbel",
            "softmax_transition_table",
            "softmax_normalize_after",
            "unbounded_h",
            "l2_normalized",
            "positive_h",
            "elu",
            "matching_network"
            ], default="softmax_normalize_after", help="short-term method")
        # TODO: for now only use weight sharing
        parser.add_argument('--generate_target_pitch', action='store_true', default=False, help="generate s_vector in fast weight frequency matrix")
    def transitions(self, batch):
        hard=self.hparams.hparams.gumbel_hard
        logits = self.slow_weight_model(batch)
        s_tar = batch
        if self.hparams.hparams.short_term_method == 'gumbel':
            if self.training:
                #h_src = torch.nn.functional.gumbel_softmax(logits, tau=self.hparams.hparams.gumbel_temperature, hard=hard, eps=1e-10, dim=2)
                #TODO: disabled hard in training. Hard is therefore only affecting evaluation.
                h_src = torch.nn.functional.gumbel_softmax(logits, tau=self.hparams.hparams.gumbel_temperature, hard=False, eps=1e-10, dim=2)
            else:
                if hard:
                    h_src = torch.nn.functional.one_hot(logits.argmax(-1), num_classes=self.hparams.hparams.encoder_output_dim).float()
                    #h_src = torch.nn.functional.gumbel_softmax(logits, tau=self.hparams.hparams.gumbel_temperature, hard=True, eps=1e-10, dim=2)
                else:
                    # TODO: READD
                    #h_src = torch.nn.functional.gumbel_softmax(logits, tau=self.hparams.hparams.gumbel_temperature, hard=False, eps=1e-10, dim=2)
                    h_src = torch.nn.functional.softmax(logits/self.hparams.hparams.gumbel_temperature, dim=-1)
        elif self.hparams.hparams.short_term_method in ["softmax_transition_table", "softmax_normalize_after"]:
            logits /= self.hparams.hparams.gumbel_temperature
            h_src = torch.nn.functional.softmax(logits, dim=2)
            if hard:
                #greedy
                if not self.training:
                    h_src = torch.nn.functional.one_hot(h_src.max(dim=-1)[1], num_classes=self.hparams.hparams.encoder_output_dim).float()
                else:
                    og_shape = h_src.shape
                    index = torch.multinomial(h_src.reshape(-1, self.m), 1).view(-1)
                    h_src_hard = torch.nn.functional.one_hot(index, num_classes=self.hparams.hparams.encoder_output_dim).float()
                    h_src_hard = h_src_hard.view(og_shape)
                    h_src = h_src_hard - h_src.detach() + h_src
        elif self.hparams.hparams.short_term_method in ['unbounded_h', "matching_network"]:
            h_src = logits
        elif self.hparams.hparams.short_term_method == 'positive_h':
            #used in combination with ordinary normalization
            h_src = torch.nn.functional.relu(logits)
        elif self.hparams.hparams.short_term_method == 'l2_normalized':
            #used in combination with ordinary normalization
            positive = torch.nn.functional.relu(logits)
            h_src = F.normalize(positive, p=2, dim=-1)
        elif self.hparams.hparams.short_term_method == 'elu':
            h_src = torch.nn.functional.elu(logits) + 1
        else:
            raise NotImplementedError('short_term method: {} is not implimented'.format(self.hparams.hparams.short_term_method))
        if self.hparams.hparams.generate_target_pitch:
            s_tar = torch.nn.functional.softmax(logits[:, :, -self.d:], dim=2)[:, 1:, :]
        return h_src, s_tar


    def probs(self, batch):
        """Returns the probabilities for full sequence

        Args:
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        probs = torch.empty(batch.shape).type_as(batch)
        h_src, s_tar = self.transitions(batch)
        #W = 1/(self.d)*torch.ones(batch.shape[0], self.m, self.d).type_as(batch)
        if self.hparams.hparams.generate_target_pitch:
            probs = matching_network(h_src, s_tar)
            return probs, None
        else:
            alpha = (self.d*100)
            W = 1/alpha * \
                torch.ones(batch.shape[0], self.m, self.d).type_as(batch)
            for i in range(batch.shape[1]):
                prob, W = step(h_src[:, i, :], s_tar[:, i, :], W, self.hparams.hparams.short_term_method)
                probs[:, i, :] = prob
            return probs, W
    
    def generate_(self, samples, W, start_i, s_tar, update_W=True, hard=False):
        with torch.no_grad():
            for i in range(start_i, samples.shape[1]):
                #TODO: avoid recalc
                h_src, _ = self.transitions(samples, hard=hard)
                prob, W_new = step(h_src[:, i, :], s_tar, W, self.hparams.hparams.short_term_method)
                W = W_new if update_W else W
                s_tar = torch.multinomial(prob, num_samples=1).squeeze(-1)
                s_tar = torch.nn.functional.one_hot(s_tar, num_classes=self.d).float()
                samples[:, i] = s_tar
            return samples

    def generate(self, init_tone, W, steps, hard=False):
        samples = torch.empty(W.shape[0], steps, self.d, dtype=torch.float32)
        s_tar = init_tone
        samples[:,0] = init_tone
            
        return self.generate_(samples, W, 1, s_tar, hard)
    def generate_prime(self, piece, steps, n_samples, update_W, hard=False):
        with torch.no_grad():
            _, W_init = self.probs(piece.unsqueeze(0))
            W_init = W_init.detach()
            shape = list(W_init.shape)
            shape[0] = n_samples
            length = piece.shape[0]
            W = torch.empty(shape, dtype=torch.float32)
            samples = torch.empty(n_samples, length + steps, piece.shape[1])
            for i in range(n_samples):
                W[i] = W_init.clone()
                samples[i,:length] = piece.detach().clone()
            s_tar = samples[:, -1]
            samples = self.generate_(samples, W, length, s_tar, hard)
            return samples
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log('annealing_temperature', torch.as_tensor(self.hparams.hparams.gumbel_temperature))
        #NOTE: Simulated anealing
        if self.hparams.hparams.simulated_annealing:
            if self.global_step % 500 == 0:
                simmulated_annealing = DSTM._create_lambda_with_globals(self.hparams.hparams.simulated_annealing)
                self.hparams.hparams.gumbel_temperature = simmulated_annealing(self.global_step)
        return loss