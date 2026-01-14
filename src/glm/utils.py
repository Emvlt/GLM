import pathlib

import numpy as np
import torch
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
