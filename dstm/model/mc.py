import torch
import torch.nn.functional as F
from pathlib import Path
import pickle
import random
import numpy as np
from dstm.util.metrics import Metrics
from dstm.util.constants import Constants
import tqdm
class MarkovChain:
    def __init__(self, d, order):
        """[summary]

        Args:
            d (int): dimension of output
            order (int): order of markov chain
        """
        self.d = d
        self.order = order
        #self.transition_matrix = torch.zeros((order + 1) * [d])    
        self.transition_matrix = torch.sparse.FloatTensor(*((order + 1) * [d]))
        #self.transition_matrix.spadd()
    def add(transition_matrix, context_index, label_index):
        indices = tuple(list(context_index) + [label_index])
        #transition_matrix[indices] = transition_matrix[indices] + 1
        #note: bug in add_ doesn't update the object
        #transition_matrix.spadd(1)
        # might be super ineficient but sparse api seems very beta, and adding 1 i
        transition_matrix = transition_matrix + torch.sparse.FloatTensor(torch.LongTensor(indices).reshape(-1,1), torch.FloatTensor([1]), transition_matrix.shape)
        return transition_matrix#= transition_matrix[indices] + 1
    def fit(self,  dataset):
        """[summary]

        Args:
            dataset (Tensor): NxTxd
        """
        for context_batch, label_index_batch in ContextLabelIterator(dataset, self.order):
            context_index_batch = context_batch.argmax(dim = -1)
            for context, label in zip(context_index_batch, label_index_batch):
                self.transition_matrix = MarkovChain.add(self.transition_matrix, context, label)
            #if self.transition_matrix.device.type is not "cpu":
            # TODO: might be needed to sum indices. For now do nothing as 
            self.transition_matrix = self.transition_matrix.coalesce()
        #samples = torch.argmax(dataset, dim = -1)
        # for piece in samples:
        #     for i in range(len(piece) - self.order):
        #        e = tuple(piece[i:(i+self.order+1)])
        #        self.transition_matrix[e] = self.transition_matrix[e] + 1 
        # Only normalize when predicting
        #self.transition_matrix = F.normalize(self.transition_matrix, p=1, dim=-1)
    def save_model(self):
        folder = 'out/model/mc'
        path = '{}/mc_o{}.pkl'.format(folder, self.order)
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump([self.d, self.order, self.transition_matrix], f)
    
    def load_model(order):
        mc = MarkovChain(0, order)
        path = 'out/model/mc/mc_o{}.pkl'.format(order)
        with open(path, 'rb') as f:
            mc.d, mc.order, mc.transition_matrix = pickle.load(f)
        return mc
    def forward1D(self, context):
        """One-step dist

        Args:
            context (Tensor): Txd
        """
        indices = torch.argmax(context, dim=-1)
        #return self.transition_matrix[tuple(indices)]
        #dense = self.transition_matrix[tuple(indices)].to_dense()
        row = self.transition_matrix[tuple(indices)].to_dense()
        normalized = F.normalize(row, p=1, dim=-1)
        return normalized
    def forward(self, context_batch):
        return torch.stack(list(map(self.forward1D, context_batch)), dim = 0)
    def predict1D(self, context):
        """One-step prediction

        Args:
            context (Tensor): orderxd
        """
        indices = torch.argmax(context, dim = -1)
        probs = self.forward1D(context)
        #TODO: bug
        if probs.sum() == 0:
            prediction = torch.Tensor([-1])
            # prediction = torch.zeros(self.d)
            # prediction[-1] = -1
        else:
            #multinomial
            prediction =  torch.multinomial(probs, 1) #.unsqueeze(0) #torch.Tensor(random.choices(np.eye(self.d), weights=dist, k=1)[0])
        return prediction

    def predict(self, context_batch):
        """Batched one-step prediction

        Args:
            context (Tensor): Bxorderxd
        """
        return torch.stack(list(map(self.predict1D, context_batch)), dim = 0)
    def nll(self, samples, reduce=True):
        """[summary]

        Args:
            samples ([type]): BxTxd

        Returns:
            [type]: [description]
        """

        #nll_loss = torch.nn.NLLLoss()
        nll = 0
        dim = samples.shape
        for context_batch, label_indices_batch in ContextLabelIterator(samples, self.order):
            prob = self.forward(context_batch) + Constants.zero_prob_add #1e-3
            logprob = torch.log(prob)
            indices = (list(range(dim[0])), label_indices_batch)
            nll -= float(logprob[indices].sum())
            #nll -= nll_loss(logprob, label_batch) 
        if reduce:
            nll /= (dim[0] * dim[1])
        print('WARNING: nll of first notes are not computed.')
        return nll
    def precision(self, samples):
        """[summary]

        Args:
            samples ([type]): BxTxd

        Returns:
            [type]: [description]
        """
        predictions = []
        labels = []
        for context_batch, label_indices_batch in ContextLabelIterator(samples, self.order):
            #note: hack for fixing return [-1,...,-1] when distribtuion is not well defined. In this case label -1 is returned
            #m, i  = self.predict(context_batch).max(-1)
            #prediction = m * i
            preiction = self.predict(context_batch)
            predictions += prediction
            labels += label_indices_batch
        print('WARNING: precision of first notes are not computed.')
        return Metrics.precision(labels, predictions)
class ContextLabelIterator:
    def __init__(self, samples, order):
        self.samples = samples
        self.order = order
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        dim = self.samples.shape 
        if self.i < dim[1]- (self.order):
            context_batch =  self.samples[:, self.i:(self.i+self.order), :]
            label_indices_batch = self.samples[:, (self.i + self.order), :].argmax(dim=-1)
            self.i += 1
            return context_batch, label_indices_batch
        else:
            raise StopIteration
class IOFitContextLabelIterator(ContextLabelIterator):
    def __init__(self, sample_batch, order, mc):
        #sample_batch = sample.unsqueeze(0)
        super().__init__(sample_batch, order)
        self.mc = mc
        self.old_context_indices = None
        self.old_label_indices = None
    def __next__(self):
        if self.old_context_indices is not None:
            #old_context_indices = self.old_context_indices.argmax(dim=-1).reshape(-1)
            self.mc.transition_matrix = MarkovChain.add(self.mc.transition_matrix, self.old_context_indices, self.old_label_indices)
        context_batch, label_indices_batch = super().__next__()

        self.old_context_indices = context_batch.argmax(-1).reshape(-1)
        self.old_label_indices = label_indices_batch.squeeze(0).reshape(-1)
        return context_batch, label_indices_batch

        
class IOMarkovChain(MarkovChain):
    def __init__(self, d, order):
        """[summary]

        Args:
            d (int): dimension of output
            order (int): order of markov chain
        """
        super().__init__(d, order)
        # self.d = d
        # self.order = order
        # self.transition_matrix = torch.zeros((order + 1) * [d])
    def nll_prec(data_loader, order):
        # nll = 0
        # predictions = []
        # labels = []
        # #B = len(data_loader) #samples.shape[0]
        # #T = data_loader[0].shape[1]
        # #d = data_loader[0].shape[2]
        # num_elem = 0
        # for batch in data_loader:
        #     sample = batch[0]
        #     d = sample.shape[2]
        #     iomc = IOMarkovChain(d, order)
        #     for context, label_index in IOFitContextLabelIterator(sample, order, iomc):
        #         #note: hack for fixing return [0,...,-1] when distribtuion is not well defined. In this case label -1 is returned
        #         #m, i  = iomc.predict(context).max(-1)
        #         #prediction = m * i
        #         prediction = iomc.predict(context)
        #         prediction = prediction.squeeze(-1)
        #         predictions.append(prediction)
        #         labels.append(label_index)
        #         prob = iomc.forward(context)
        #         p = prob[0, label_index]
        #         p = p if p> 0 else torch.Tensor([Constants.log_zero_prob])
        #         nll -= torch.log(p)
        #         num_elem +=1
        # precision = Metrics.precision(labels, predictions)
        # return nll/ num_elem, precision
        _, logprops, tps = IOMarkovChain.predict_all(data_loader, order)
        return -logprops.mean(), tps.float().mean()

    def forward(self, context_batch):
        row = super().forward(context_batch)
        return F.normalize(row, p=1)
    def predict_all(data_loader, order):
        predictions = []
        tps = []
        logprobs = []
        labels = []
        for batch in tqdm.tqdm(data_loader):
            sample = batch[0]
            d = sample.shape[2]
            iomc = IOMarkovChain(d, order)
            if sample.shape[1] == 0:
                continue
            for _ in range(order):
                predictions.append(torch.tensor(-1))
                tps.append(False)
                logprobs.append(np.log(Constants.zero_prob_add))
            for context, label_index in IOFitContextLabelIterator(sample, order, iomc):
                prediction = iomc.predict(context)
                prediction = prediction.squeeze(-1)
                predictions.append(prediction)
                labels.append(label_index)
                tps.append(prediction == label_index)
                prob = iomc.forward(context)
                p = prob[0, label_index]
                p = p if p> 0 else torch.Tensor([Constants.zero_prob_add])
                logprobs.append(torch.log(p))
        
        return torch.tensor(predictions), torch.tensor(logprobs), torch.tensor(tps)

                

