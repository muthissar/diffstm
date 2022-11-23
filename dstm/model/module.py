import torch
from dstm.util.metrics import Metrics
import pytorch_lightning as pl
from argparse import ArgumentParser
from dstm.util.constants import Constants
import sys
from dstm.model.coded_short_term_model import *
import tqdm

class Module(pl.LightningModule):
    def __init__(self, d, hparams):
        """
        Args:
            d ([type]): [description]
            hparams (obj): namespace of hyper parameters
        """
        super().__init__()
        if hparams.no_one_hot:
            def general_cross_entropy(logprobs, labels):
                return -(logprobs * labels).mean()
            self.loss = general_cross_entropy
        else:
            self.loss = torch.nn.NLLLoss(reduction='mean')
        self.d = d
        self.lr = hparams.lr
        self.save_hyperparameters('d', 'hparams')
    @staticmethod
    def add_model_specific_args(parent_parser, subparsers):
        parser = ArgumentParser(description="model.module.Model", parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
        parser.add_argument('--l2_regularization_strength', type=float, default=0.0, help="use encoder l2 regularization")
        parser.add_argument('--dropout', type=float, default=0.0, help="use dropout")
        parser.add_argument('--no_one_hot', action='store_true', default=False, help="(experimental) data is not one-hot encoded")
        parser.add_argument('--encoder', type=str, choices=["transformer_rel", "transformer_lin", "transformer_abs", "dccnn"], default="dccnn", help="encoder model to be used")
        #TODO should only be enabled when dcccn
        parser.add_argument('--filters', type=int, nargs="*",
                            default=[1024, 1024, 1024, 1024, 1024, 1024])
        parser.add_argument('--activation', type=str, choices=["relu", "selu", "gated_activation", "none"], default="selu", help="activation to be used for dccnn")
        #TODO: should only be enabled when transformer is used
        parser.add_argument('--transformer_n_head', type=int, default=7, help="number of heads to be used for multi-head attention in transformer encoders")
        parser.add_argument('--transformer_layers', type=int, default=6, help="number of causal transformer layers to be used")
        parser.add_argument('--encoder_output_dim', type=int, default=512, help="encoder output dimensionality")
        return parser
    
    def probs(self, batch):
        raise NotImplementedError("Abstract method")
    def shared_step(self, batch):
        if self.hparams.hparams.no_one_hot:
            labels = batch[0]
        else:
            labels = batch[0].argmax(dim=-1)
        probs, _ = self.probs(batch[0])
        #In case of zero padding for differing lengths
        #TODO: could be more efficient with a mask...
        if len(batch) == 2:
            lengths = batch[1]
            labels_flattened_list = []
            probs_flattened_list = []
            for label, prob, length in zip(labels, probs, lengths):
                labels_flattened_list.append(label[:length])
                probs_flattened_list.append(prob[:length])
            labels_flattened = torch.cat(labels_flattened_list)
            probs_flattened = torch.cat(probs_flattened_list)
        else:
            labels_flattened = labels.reshape(-1)
            probs_flattened = probs.reshape(-1, self.d)
        #Add constant for zero prob
        probs_flattened += Constants.zero_prob_add*(probs_flattened == 0)
        logprobs_flattened = torch.log(probs_flattened)
        loss = self.loss(logprobs_flattened, labels_flattened)
        return probs_flattened, loss, labels_flattened

    def forward(self, batch):
        return self.shared_step(batch)
    def training_step(self, batch, batch_idx):
        _, loss, _ = self.shared_step(batch)
        self.log('train_loss', loss)
        grad_abs_max = -1
        for param in filter(lambda x: x.requires_grad, self.parameters()):
            m = torch.max(torch.abs(param))
            if m > grad_abs_max:
                grad_abs_max = m
        # sch = self.lr_schedulers()
        # if sch is not None:
        #     sch.step()
        self.log('grad_abs_elem_max', grad_abs_max,
                 on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq_length = batch[0].shape[1]
        if seq_length == 0:
            return torch.FloatTensor([]).type_as(batch[0]), torch.LongTensor([])
        seq_max_length = self.hparams.hparams.seq_max_length
        #NOTE in this case batch_size is 1 for transformers
        if seq_length > seq_max_length and self.hparams.hparams.encoder != 'dccnn':
            probs_batch = []
            labels_batch = []
            for i in range(seq_max_length, seq_length + 1):
                #lengths = [max(0,min(seq_max_length, l-i)) for l in batch[1]]
                batch_small = batch[0][:, i-seq_max_length:i, :]
                labels = batch[0][:, i-seq_max_length:i, :].argmax(dim=-1)
                probs, _ = self.probs(batch_small)
                if i == seq_max_length:
                    probs_batch.append(probs)
                    labels_batch.append(labels)
                    #break
                else:
                    probs_batch.append(probs[:, -1:])
                    labels_batch.append(labels[:, -1:])
            probs_batch = torch.cat(probs_batch, dim=1)
            labels_batch = torch.cat(labels_batch, dim=1)
            probs_flattened = []
            labels_flattened = []
            for length, prob, label in zip(batch[1], probs_batch, labels_batch):
                probs_flattened.append(prob[:length])
                labels_flattened.append(label[:length])
            probs_flattened = torch.cat(probs_flattened)
            labels_flattened = torch.cat(labels_flattened)
        else:
            probs_flattened, _, labels_flattened = self.shared_step(batch)
        return probs_flattened, labels_flattened
    

    def validation_epoch_end(self, outputs):
        unziped =  list(zip(*outputs))
        probs_flattened = torch.cat(unziped[0])
        labels_flattened = torch.cat(unziped[1])
        if self.hparams.hparams.no_one_hot:
            probs_flattened = probs_flattened
        else:
            predictions = probs_flattened.argmax(dim=-1)
        probs_flattened += Constants.zero_prob_add*(probs_flattened == 0)
        logprobs_flattened = torch.log(probs_flattened)
        loss = self.loss(logprobs_flattened, labels_flattened)
        precision, tp_std, num_obs = Metrics.tp_stats(labels=labels_flattened.cpu(), predictions=predictions.cpu())
        
        self.log('val_precision', precision, sync_dist=True)
        self.log('tp_std', tp_std, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)

    def train_batch_end(self, trainer, pl_module, outputs):
        super().on_train_batch_end(trainer, pl_module, outputs)  # don't forget this :)
        percent = (self.train_batch_idx / self.total_train_batches) * 100
        sys.stdout.flush()

    def configure_optimizers(self):
        #Only add l2 reg to weihgt parametersq
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.hparams.hparams.l2_regularization_strength
        )
        #NOTE: let's try SWA Instead...
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=3.5e-4,
        #     epochs=10,
        #     steps_per_epoch=2400
        # )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=20
        #     )
        # lr_scheduler_config = {
        #     'scheduler': scheduler,
        #     'interval': "epoch",
        #     'frequency': 1,
        #     'name': None

        # }
        #return [optimizer], [lr_scheduler_config]
        return optimizer

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights		            
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]