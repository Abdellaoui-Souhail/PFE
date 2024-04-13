"""
Helpers for distributed training.
"""

import torch as th

GPUS_PER_NODE = 1

def setup_dist():
    """
    No setup needed for a single GPU setup.
    """
    pass

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda:0")  # Assuming you have just one GPU
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)

def sync_params(params):
    """
    No synchronization needed for a single GPU setup.
    """
    pass

