import torch
def delete_param(p, in_path, out_path=False):
    obj = torch.load(in_path)
    hparams = obj['hyper_parameters']['hparams']
    del vars(hparams)['p']
    if out_path:
        torch.save(obj, out_path)
def save_param(p, val, in_path, out_path=False):
    obj = torch.load(in_path)
    hparams = obj['hyper_parameters']['hparams']
    vars(hparams)[p] = val
    #obj['hyper_parameters']['hparams'] = hparams
    if out_path:
        torch.save(obj, out_path)