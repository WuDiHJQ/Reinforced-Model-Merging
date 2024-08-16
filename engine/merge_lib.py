import copy
import torch
from collections import OrderedDict


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def rescale_sum(tensor: torch.Tensor, mask: torch.Tensor):
    """Rescales the values to match the original tensor sum."""
    org_sum = tensor.abs().sum()
    new_sum = (tensor * mask).abs().sum()

    if org_sum >= 1e-8 and new_sum >= 1e-8:
        tensor *= org_sum / new_sum
    return tensor * mask


def magnitude(tensor: torch.Tensor, density: float, rescale: bool):
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.numel())

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.argsort(w, descending=True)[:k]
    mask.view(-1)[topk] = 1

    if rescale:
        res = rescale_sum(tensor, mask)
    else:
        res = tensor * mask

    return res


def bernoulli(tensor: torch.Tensor, density: float, rescale: bool):
    if density >= 1:
        return tensor

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
    )
    res = tensor.to(work_dtype) * mask
    if rescale:
        res /= density

    return res.to(tensor.dtype)


def GeneralizedTaskArithmetic(consensus_method, sparsification_method, rescale, density, base_ckpt, pt_ckpt):

    weight = 0.3
    remove_keys = []
    flat_base = [state_dict_to_vector(ckpt, remove_keys) for ckpt in base_ckpt]
    flat_pt = state_dict_to_vector(pt_ckpt, remove_keys)

    tvs = get_task_vectors(flat_base, flat_pt)

    # sparsify
    if sparsification_method:
        for tv in tvs:
            if sparsification_method == "magnitude":
                tv["delta"] = magnitude(tv["delta"], density=density, rescale=rescale)
            elif sparsification_method == "random":
                tv["delta"] = bernoulli(tv["delta"], density=density, rescale=rescale)

    deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)

    weights = torch.stack([torch.full(tv["delta"].shape, weight, dtype=deltas.dtype, device=deltas.device) for tv in tvs], dim=0)
    
    weighted_deltas = deltas * weights

    # get sign consensus and mix deltas
    if consensus_method:
        mask_dtype = flat_pt.dtype
        mask = get_mask(
            weighted_deltas,
            method=consensus_method,
            mask_dtype=mask_dtype,
        )
        mixed_delta = (weighted_deltas * mask).sum(dim=0)
    else:
        mixed_delta = weighted_deltas.sum(dim=0)

    return vector_to_state_dict((flat_pt + mixed_delta).to(flat_pt.dtype), pt_ckpt, remove_keys)


def get_task_vectors(flat_base, flat_pt):
    tvs = []
    for base in flat_base:
        delta = base - flat_pt
        d = {"delta":delta}
        tvs.append(d)
    return tvs


def get_mask(delta, method, mask_dtype):
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign


def task_arithmetic(base_ckpt, pt_ckpt):  
    return GeneralizedTaskArithmetic(consensus_method=None, 
                                 sparsification_method=None, 
                                 rescale=False, 
                                 density=0.3,
                                 base_ckpt=base_ckpt, 
                                 pt_ckpt=pt_ckpt)


def ties(base_ckpt, pt_ckpt):
    return GeneralizedTaskArithmetic(consensus_method='sum', 
                                     sparsification_method='magnitude', 
                                     rescale=False, 
                                     density=0.3,
                                     base_ckpt=base_ckpt, 
                                     pt_ckpt=pt_ckpt)
    
    
def dare(base_ckpt, pt_ckpt):
    return GeneralizedTaskArithmetic(consensus_method=None, 
                                     sparsification_method='random', 
                                     rescale=True, 
                                     density=0.66,
                                     base_ckpt=base_ckpt, 
                                     pt_ckpt=pt_ckpt)    
    
    
def dare_ties(base_ckpt, pt_ckpt):
    return GeneralizedTaskArithmetic(consensus_method='sum', 
                                     sparsification_method='random', 
                                     rescale=True, 
                                     density=0.66,
                                     base_ckpt=base_ckpt, 
                                     pt_ckpt=pt_ckpt)