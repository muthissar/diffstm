import torch
import torch.nn.functional as F

def step(h_src, s_tar, W, method):
    if method in ["softmax_transition_table"]:
    #if method in ["softmax_transition_table", "gumbel"]:
        return step_transition_table(h_src, s_tar, W)
    elif method in ["gumbel", "softmax_normalize_after", "l2_normalized", "positive_h", "generate_target_pitch_shared_weights", "elu"]:
    #elif method in ["softmax_normalize_after", "l2_normalized", "positive_h"]:
        return step_normalize_after(h_src, s_tar, W)
    elif method in  ["unbounded_h"]:
        return step_unbounded_h(h_src, s_tar, W)
    else:
        raise NotImplementedError("Short Term Method Unkown.")

def step_transition_table(h_src, s_tar, W):
    """[summary]
    Args:
        h_src (Tensor): Bxm
        s_tar (Tensor): Bxd
        W (Tensor): Bxmxd
    Returns:
        [type]: [description]
    """
    # predict using old W
    W_normed = F.normalize(W, p=1, dim=2)
    pred = torch.bmm(h_src.unsqueeze(1), W_normed)
    pred = pred.squeeze(1)
    # update using W
    W = W + torch.bmm(h_src.unsqueeze(2), s_tar.unsqueeze(1))
    return pred, W


def step_normalize_after(h_src, s_tar, W):
    """[summary]
    Args:
        h_src (Tensor): Bxm
        s_tar (Tensor): Bxd
        W (Tensor): Bxmxd
    Returns:
        [type]: [description]
    """
    # predict using old W
    #print(torch.bmm(h_src.unsqueeze(1), W).min(), torch.bmm(h_src.unsqueeze(1), W).max())
    pred = F.normalize(torch.bmm(h_src.unsqueeze(1), W), p=1, dim=2)
    pred = pred.squeeze(1)
    #del W_normed
    # update using W
    W = W + torch.bmm(h_src.unsqueeze(2), s_tar.unsqueeze(1))
    return pred, W

def step_unbounded_h(h_src, s_tar, W):
    #TODO: this does prboably not work to sum dot products and use softmax (exponential)
    """[summary]
    Args:
        h_src (Tensor): Bxm
        s_tar (Tensor): Bxd
        W (Tensor): Bxmxd
    Returns:
        [type]: [description]
    """
    # predict using old W
    # For scaled dot product transformer...
    #pred = F.softmax(1/h_src.shape[1] *torch.bmm(h_src.unsqueeze(1), W), dim=2)
    pred = F.softmax(torch.bmm(h_src.unsqueeze(1), W), dim=2)
    #CHANGED
    pred = pred.squeeze(1)
    #del W_normed
    # update using W
    W = W + torch.bmm(h_src.unsqueeze(2), s_tar.unsqueeze(1))
    return pred, W

def matching_network(h_srcs, s_tars):
    shape = list(s_tars.shape)
    shape[1] += 1
    probs = torch.empty(shape).type_as(h_srcs)
    # todo: for now set the prior uniformly
    probs[:, 0, :] = 1/shape[2] 
    for i in range(1, shape[1]):
        dot_product = h_srcs[:, :i :].bmm(h_srcs[:, i, :].unsqueeze(2))
        softmax = F.softmax(dot_product.permute([0, 2, 1]),dim=-1)
        probs[:, i, :] = softmax.bmm(s_tars[:,:i,:]).squeeze(1)
    return probs