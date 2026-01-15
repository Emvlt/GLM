from typing import Dict, List
import pathlib
import math

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class ScanDataset(Dataset):
    def __init__(
        self,
        path_to_dataset:pathlib.Path,
        data_tuples : list,
        mode:str,
        training_proportion=0.8,
        validation_proportion=0.1,
        testing_proportion=0.1
        ):
        ### Defining the backend
        ### Defining the path to data
        self.path_to_dataset = path_to_dataset
        """
        The path_to_dataset attribute (pathlib.Path) points towards the folder
        where the data is stored
        """
        ### Defining the path to the data record
        self.path_to_data_record = pathlib.Path(f'src/glm/default_data_records.csv')
        ### Defining the data record
        self.data_record = pd.read_csv(self.path_to_data_record)
        """
        The data_record (pd.Dataframe) maps a slice index/identifier to
            - the sample index of the sample it belongs to
            - the number of slices expected
            - the number of slices actually sampled
            - the first slice of the sample to which a given slice belongs
            - the last slice of the sample to which a given slice belongs
            - the mix (if indicated)
            - the detector it was sampled with
        """
        self.data_tuples = data_tuples
        
        assert training_proportion + validation_proportion + testing_proportion == 1
        self.training_proportion = training_proportion
        """
        The training_proportion (float) argument defines the proportion of the dataset used for training
        """
        self.validation_proportion = validation_proportion
        """
        The validation_proportion (float) argument defines the proportion of the dataset used for validation
        """
        self.testing_proportion = testing_proportion
        """
        The testing_proportion (float) argument defines the proportion of the dataset used for testing
        """
        ### We split the dataset between training and testing
        """
        The sample_dataframe (pd.Dataframe) is a dataframe linking sample index to slices.
        It is used to partition the dataset on a sample basis, rather than on a slice basis,
        avoiding 'data leakage' between training and testing
        """
        n_samples = len(self.data_record['sample_index'].unique())

        if mode == 'training':
            first_sample_index = 0
            last_sample_index  = math.floor(n_samples*self.training_proportion) 

        elif mode == 'validation':
            first_sample_index = math.floor(n_samples*self.training_proportion)
            last_sample_index  = math.ceil(n_samples*(self.training_proportion+self.validation_proportion))  

        elif mode == 'testing':
            first_sample_index = math.ceil(n_samples*(self.training_proportion+self.validation_proportion))
            last_sample_index  = n_samples 

        else:
            raise NotImplementedError
        self.slice_dataframe = self.data_record.query(f'{first_sample_index} < sample_index <= {last_sample_index}')

    def __len__(self):
        return len(self.slice_dataframe)
    
    def load_data(
        self,
        slice_row,
        modality,
        mode,
        tensor_dict  :Dict[str, torch.Tensor],
        ):
        path_to_folder = self.path_to_dataset.joinpath(f"{slice_row['slice_identifier']}/{mode}")
        if modality in [
            'preprocessed_reconstruction', 
            'preprocessed_sinogram'
            ]:
            data = torch.from_numpy(np.load(path_to_folder.joinpath(f'{modality}.npy'))
                                    ).unsqueeze(0)
        else:
            raise NotImplementedError
        
        tensor_dict[f'{modality}_{mode}'] = data

        return tensor_dict    

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        slice_row = self.slice_dataframe.iloc[index]
        tensor_dict = {}
        for (modality, mode) in self.data_tuples:
            tensor_dict = self.load_data(
                slice_row,
                modality,
                mode, 
                tensor_dict
            )

        return tensor_dict
    
def parse_dataloader(
        dataset_path : str,
        mode:str,
        data_tuples:List[tuple],
        batch_size : int, 
        num_workers : int,
        distributed=False,
        rank=0,
        world_size=1        
        ):
    
    dataset = ScanDataset(
        path_to_dataset = pathlib.Path(dataset_path),
        data_tuples = data_tuples,
        mode = mode,
    )

    print(f'Instanciating 2detect {mode} dataset with size {len(dataset)}')

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(mode == 'training')
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = (mode == 'training')

    return DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        sampler = sampler,
        pin_memory=True,
        drop_last=True  # Important for consistent batch sizes
        )