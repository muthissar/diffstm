import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dstm.util.load_data import SessionPreprocessing
from dstm.util.constants import Constants
from scipy.stats import norm
import numpy as np
import tqdm
class BaseLine:
    # abstract model
    #pre_processing = {}
    train_loaders = {}
    test_loaders = {}
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset in BaseLine.train_loaders:
            pass
        elif dataset == "session":
            pre_processing = SessionPreprocessing(loop=False)
            BaseLine.train_loaders[self.dataset] = pre_processing.get_data_loader('train', shuffle=False)
            BaseLine.test_loaders[self.dataset] = pre_processing.get_data_loader('test', shuffle=False)
        else:
            raise NotImplementedError("dataset {} is not implimented".format(dataset))
    @property
    def train_loader(self):
        if self.dataset in BaseLine.train_loaders:
            pass
        elif self.dataset == "session":
            preprocessing = SessionPreprocessing(loop=False)
            BaseLine.train_loaders[self.dataset] = preprocessing.get_data_loader('train', shuffle=False)
        return BaseLine.train_loaders[self.dataset]

    @property
    def test_loader(self):
        if self.dataset in BaseLine.test_loaders:
            pass
        elif self.dataset == "session":
            preprocessing = SessionPreprocessing(loop=False)
            BaseLine.test_loaders[self.dataset] = preprocessing.get_data_loader('test', shuffle=False)
        return BaseLine.test_loaders[self.dataset]

    def get_model_folder(self):
        folder = 'out/{}/model/{}'.format(self.dataset, self.model)
        Path(folder).mkdir(parents=True, exist_ok=True)
        return folder

class Prior(BaseLine):
    def __init__(self, **kwargs):
        self.model = "Prior"
        super().__init__(**kwargs)
        self.model_file = "{}/{}".format(self.get_model_folder(), "Prior.p")
    def _save_model(self):
        torch.save(self.params, self.model_file)
    def fit(self):
        data = torch.cat([self.train_loader.dataset]).argmax(-1).tolist()
        d = self.train_loader.dataset[0].shape[-1]
        self.params = plt.hist(data, bins=list(range(0, d+1)), align='left', density=True)[0]
        self._save_model()
    def load_model(self):
        self.params = torch.load(self.model_file)
    def nll(self):
        d = self.test_loader.dataset[0].shape[-1]
        data = torch.cat([*self.test_loader.dataset]).view(-1, d)
        logprobs = torch.log(torch.FloatTensor(self.params)).view(d, 1)
        return - (data.mm(logprobs)).mean()
class IOPrior(BaseLine):
    def __init__(self, **kwargs):
        self.model = "PriorIO"
        super().__init__(**kwargs)
    def predict(self):
        d = self.test_loader.dataset[0].shape[-1]
        tps = []
        logprobs = []
        predictions = []
        for x in self.test_loader.dataset:
            if len(x) == 0:
                continue
            predictions.append(-1)
            tps.append(False)
            hist = torch.zeros(d)
            hist[x[0].argmax()] = 1
            logprobs.append(np.log(Constants.zero_prob_add))
            for i, symbol in enumerate(x[1:], 1):
                #piece = x.argmax(-1)
                #hist = np.histogram(piece[:i], bins=list(range(0, d+1)), density=True)[0]
                #logprob = torch.log(x.mm(torch.FloatTensor(hist).view(d, 1)))
                prop = hist[hist.argmax(-1)]/hist.sum(-1)
                logprob = np.log(prop) if prop > 0 else np.log(Constants.zero_prob_add)
                logprobs.append(logprob.item())
                prediction = hist.argmax(-1)
                hist[symbol.argmax(-1)] += 1
                predictions.append(prediction.item())
                tps.append((symbol.argmax(-1) == prediction).item())
        return predictions, logprobs, tps
    def nll_precision(self):
        _, logprobs, tps = self.predict()
        return - torch.tensor(logprobs).mean(), torch.tensor(tps).float().mean()
class Repetition(BaseLine):
    def __init__(self, **kwargs):
        self.model = "Repetition"
        super().__init__(**kwargs)
    def predict(self):
        tps = []
        predictions = []
        logprobs = []
        for x in tqdm.tqdm(self.test_loader.dataset):
            if len(x) == 0:
                continue
            piece = x.argmax(-1)
            prediction = torch.zeros_like(piece)
            prediction[0] = -1
            prediction[1:] = piece[:-1]
            predictions.append(prediction)
            tp = prediction == piece
            tps.append(tp.float())
            #result = piece[:-1] == piece[1:]
            result = (~tp)* Constants.zero_prob_add + (tp)
            #d = x.shape[-1]
            #eps = 1e-7
            #result = (~result)*(1-(1/d+eps))/(d-1) + (1/d + eps )*(result)
            logprobs.append(torch.log(result))

        return torch.cat(predictions), torch.cat(logprobs), torch.cat(tps)
    def nll(self):
        logprobs = []
        for x in self.test_loader.dataset:
            piece = x.argmax(-1) 
            result = piece[:-1] == piece[1:]  
            result = (~result)* Constants.zero_prob_add + (result)
            logprobs.append(torch.log(result))
        return - torch.cat(logprobs).mean()
    def precision(self):
        # tps = []
        # for x in self.test_loader.dataset:
        #     piece = x.argmax(-1) 
        #     result = piece[:-1] == piece[1:]  
        #     tps.append(result.float())
        _, tps = self.predict()
        #return torch.cat(tps).mean() 
        return tps.mean()        

def confidence_interval(p, n):
    bound = norm.ppf(0.975, loc=0, scale=1)*np.sqrt(p*(1-p)/n)
    return p - bound, p + bound