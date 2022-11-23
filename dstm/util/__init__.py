#import yaml
import torch
import numpy as np
# def get_config():
#     conf = None
#     with open('config.yml', 'r') as file:
#         conf = yaml.safe_load(file)
#     return conf

def entropy(h_src):
    return - (h_src * torch.log(h_src)).sum(-1)
def efficiency(h_src):
    m = h_src.shape[-1]
    ent = entropy(h_src)
    return ent/np.log(m)
    #return 

def add_centered(t1, t2, pitch_relative_to, time_relative_to, relative_pitch=True):
    d = t2.shape[-1]
    r = t1.shape[-2]
    if relative_pitch:
        pitch_index = slice(d-pitch_relative_to-1, 2*d - pitch_relative_to-1)
    else:
        pitch_index = slice(None)
    t1[max((r-time_relative_to), 0):, pitch_index] += t2[max(0, time_relative_to-r):time_relative_to, :]
def piano_to_pitch_change(label, probs):
    changes = torch.diff(label, prepend=torch.tensor([-1], device = label.device)) != 0
    return probs[changes], label[changes]