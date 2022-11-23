import torch
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector
import numpy as np
from dstm.util.load_data import EssenPreprocessing, SessionPreprocessing
from dstm.util.metrics import Metrics
import copy
from pathlib import Path
import pickle
import functools
import tqdm.contrib.concurrent
import tqdm
ppm = importr("ppm")
EPOCHS = 1
LOOP_DATA = False
class PPMTrainer:
    def __init__(self, type_="simple", io=True, dataPreprocessor=None,**kwargs):
        self.dataPreprocessor = dataPreprocessor
        self.global_time = 1
        #maybe max order should be receptive field of compared model
        self.type = type_
        self.io = io
        alphabet_size = self.dataPreprocessor.d
        shared_args = {"alphabet_size":alphabet_size, **kwargs}
        if type_ == "simple":
            self.model_gen = lambda: ppm.new_ppm_simple(**shared_args)
        elif type_ == "decay":
            self.model_gen = lambda: ppm.new_ppm_decay(**shared_args, ltm_half_life = 2)
        else:
            raise NotImplementedError("type is not allowed")
        self.model = self.model_gen()
        
    def one_hot_to_r(self, seq):
        return (1 + seq.argmax(dim=-1)).tolist()
    def forward(self, batch):
        nll = []
        sum_of_lengths = 0
        preds = []
        labels = []
        if self.io:
            self.model = self.model_gen()
        for seq, length in zip(batch[0], batch[1]):
            seq_r = self.one_hot_to_r(seq[:length])
            if self.type == "decay":
                #time = [i + 0.0 for i in range(1,length + 1)]
                time = list(range(self.global_time, self.global_time + length))
                self.global_time += length
                res = ppm.model_seq(model=self.model, seq=seq_r, time=IntVector(time), train=True)
                ic = res[4]
                props = res[6]
            elif self.type == "simple":
                res = ppm.model_seq(model=self.model, seq=seq_r, train=True)
                ic = res[2]
                props = res[4]
            #information content
            nll += ic
            sum_of_lengths += length
            labels += seq_r
            #hard (use max)
            #preds += (torch.multinomial(torch.tensor(np.array(props)), 1) +1).view(-1).tolist() 
            preds += list(np.array(props).argmax(axis=-1)+1)
        return nll, preds, labels, sum_of_lengths
    def train(self):
        batch_size = 16
        train_loader = self.dataPreprocessor.get_data_loader('train',
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
        )
        for _ in range(EPOCHS):
            for i, batch in enumerate(train_loader):
                nll, _, _, sum_of_lengths = self.forward(batch)
                print("train step {}/{}, nll: {}".format(i, len(train_loader.dataset)//batch_size+ 1, nll/sum_of_lengths))
    def predict_all(self, split, max_workers):
        nll_all = []
        sum_of_lengths_all = 0
        preds_all = []
        labels_all = []
        loader = self.dataPreprocessor.get_data_loader(split,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )

        for i, batch in tqdm.tqdm(enumerate(loader)):
            if np.prod(batch[0].shape):
                nll, preds, labels, sum_of_lengths = self.forward(batch)
                nll_all += nll
                sum_of_lengths_all += sum_of_lengths
                preds_all += preds
                labels_all += labels
            else:
                print("No elements at position {}".format(i))
        return preds_all, nll_all, labels_all, sum_of_lengths_all
        # def proc_one(i, batch):
        #     if np.prod(batch[0].shape):
        #         #nll, preds, labels, sum_of_lengths = self.forward(batch)
        #         return self.forward(batch)
        #     else:
        #         print("No elements at position {}".format(i))
        # # nll, preds, labels, sum_of_lengths =  zip(*tqdm.contrib.concurrent.process_map(proc_one, range(len(loader)), loader))
        # # preds_all = functools.reduce(lambda a, b: a+b, nll)
        # # nll_all = functools.reduce(lambda a, b: a+b, preds)
        # packed_res = zip(*tqdm.contrib.concurrent.process_map(proc_one, range(len(loader)), loader), max_workers=max_workers)
        # reduced = map(lambda x: functools.reduce(lambda a, b: a+b, x), packed_res)
        # return reduced

    def validate(self, split):
        preds_all, nll_all, labels_all, sum_of_lengths_all = self.predict_all(split, max_workers=None)
        return sum(nll_all)/sum_of_lengths_all, *Metrics.tp_stats(labels=labels_all, predictions=preds_all) #Metrics.precision(labels=labels_all, predictions=preds_all)
    