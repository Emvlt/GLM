import pathlib
import os 

import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

def _process_numpy(data:np.ndarray) -> np.ndarray:
    return data

def _process_torch(data:torch.Tensor) -> np.ndarray:
    if len(data.shape) == 3:
        return data[0].detach().cpu().numpy()
    
    if len(data.shape) == 4:
        return data[0,0].detach().cpu().numpy()
    
    raise ValueError(f'Not implemented for data shape {data.shape} of dim {len(data.shape)}')

def plot_image(data, savepath, extension=None, title=None):
    if extension is None:
        extension = 'jpg'

    if isinstance(data, np.ndarray):
        data = _process_numpy(data)
    elif isinstance(data, torch.Tensor):
        data = _process_torch(data)
    else:
        raise NotImplementedError
    
    savepath = pathlib.Path(savepath)
    savepath.parent.mkdir(exist_ok=True, parents=True)
    
    plt.matshow(data)
    if title is not None and isinstance(title, str):
        plt.title(title)
    plt.colorbar()
    plt.savefig(f'{savepath}.{extension}', bbox_inches = 'tight')
    plt.clf()

def plot_image_live(
        data, live_session, name, extension=None, title=None
        ):
    if extension is None:
        extension = 'jpg'

    if isinstance(data, np.ndarray):
        data = _process_numpy(data)
    elif isinstance(data, torch.Tensor):
        data = _process_torch(data)
    else:
        raise NotImplementedError
    
    fig, axs = plt.subplots()
    axs.matshow(data)
    if title is not None and isinstance(title, str):
        axs.set_title(title)
    live_session.log_image(f"{name}.{extension}", fig)

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("WARNING: distributed environment variables not found. Defaulting to single process.")
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        print(f'Initialising process group on device {local_rank}')
        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        # initialize the process group
        dist.init_process_group(backend, rank=rank, world_size=world_size, init_method='env://')
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
