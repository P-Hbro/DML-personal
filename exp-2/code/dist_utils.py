import os
import torch
import torch.nn as nn
import torch.distributed as dist


def dist_init(world_size, rank):
    """
    Init the pytorch distributed communication lib
    Args:
        rank: (int) the device ID
        world_size: (int) the total devices
    """
    # change it to the corresponding ip addr
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    assert dist.is_initialized(), "Error! The distributed env is not initialized!"

    return True


def cleanup():
    """
    destroy the communication group
    """
    dist.destroy_process_group()


def get_local_rank():
    """
    get the local rank (devices id)
    """
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def init_parameters(model):
    """
    Boradcast the initial gradients of the model parameters、
    """
    if get_world_size() > 1:
        # implement your own init
        for param in model.parameters():
            dist.broadcast(param.data, 0)



# 1. all_reduce
def allreduce_average_gradients(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        # implement your own aggregation method
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def allgather_average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        all_gradients = [torch.zeroslike(param.grad.data) for _ in range(size)]
        dist.all_gather(all_gradients, param.grad.data)
        param.grad.data = torch.stack(all_gradients, dim=0).mean(dim=0)