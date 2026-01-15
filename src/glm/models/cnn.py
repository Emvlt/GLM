from typing import Dict 

import torch
import torch.nn as nn

class CNN_Module(nn.Module):
    def __init__(
        self,
        input_channels : int, 
        out_channels:int,
        parameters_dict : Dict
        ):
        super().__init__()

        self.dimension = parameters_dict['kernel_dimension'] 
        kernel_size = parameters_dict['kernel_size'] 
        padding = parameters_dict['padding']

        if self.dimension == 2:
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channels, out_channels, kernel_size, padding=padding, padding_mode='reflect'),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, padding_mode='reflect'),
                nn.ReLU()
            )

        else:
            raise ValueError
            
    def forward(self, input_tensor:torch.Tensor):
        assert len(input_tensor.size()) == 2 + self.dimension
        out = self.conv1(input_tensor)
        return self.conv2(out) + out
    
    def set_data_shape(self, 
                       bs, angles_indices, 
                       n_measurements,
                       n_pixels,
                       input_tensor:torch.Tensor, target:str):
        if angles_indices is not None:
            input_tensor = input_tensor[:,:,angles_indices, :]
        if target == 'NN':
            assert len(input_tensor.size()) == 4
            if self.dimension == 1:
                return input_tensor.view(bs*n_measurements, 1, n_pixels)
            elif self.dimension == 2 : 
                return input_tensor
            
        elif target == 'tomo':
            assert len(input_tensor.size()) == 2 + self.dimension
            if self.dimension == 1:
                return  input_tensor.view(bs, n_measurements, 1 ,n_pixels) 
            elif self.dimension == 2 : 
                return input_tensor

        elif target == 'display':
            assert len(input_tensor.size()) == 2 + self.dimension
            if self.dimension == 1:
                return  input_tensor.view(bs, n_measurements, 1 ,n_pixels) 
            elif self.dimension == 2 : 
                return input_tensor
        else:
            raise ValueError(f'Got {target} but expected NN, tomo or display')
        
class ImageCNN(nn.Module):
    def __init__(
        self,
        input_channels : int, 
        out_channels:int,
        parameters_dict : Dict
    ):
        super().__init__()
        self.dimension = parameters_dict['kernel_dimension'] 
        kernel_size = parameters_dict['kernel_size'] 
        padding = parameters_dict['padding']
        n_channels = parameters_dict['n_channels']
            
        if self.dimension == 2:
            self.model = nn.Sequential(
                nn.Conv2d(input_channels, n_channels, kernel_size, padding=padding),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(n_channels, n_channels, kernel_size, stride=1, padding=padding),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(n_channels, out_channels, kernel_size, stride=1, padding=padding),
                nn.LeakyReLU(negative_slope=0.1),
            )

        else:
            raise ValueError
            
    def forward(self, input_tensor:torch.Tensor):
        assert len(input_tensor.size()) == 2 + self.dimension
        return self.model(input_tensor)
    
    def set_data_shape(self, 
                       bs, angles_indices, 
                       n_measurements,
                       n_pixels,
                       input_tensor:torch.Tensor, target:str):
        if angles_indices is not None:
            input_tensor = input_tensor[:,:,angles_indices, :]
        if target == 'NN':
            assert len(input_tensor.size()) == 4
            if self.dimension == 1:
                return input_tensor.view(bs*n_measurements, 1, n_pixels)
            elif self.dimension == 2 : 
                return input_tensor
            
        elif target == 'tomo':
            assert len(input_tensor.size()) == 2 + self.dimension
            if self.dimension == 1:
                return  input_tensor.view(bs, n_measurements, 1 ,n_pixels) 
            elif self.dimension == 2 : 
                return input_tensor

        elif target == 'display':
            assert len(input_tensor.size()) == 2 + self.dimension
            if self.dimension == 1:
                return  input_tensor.view(bs, n_measurements, 1 ,n_pixels) 
            elif self.dimension == 2 : 
                return input_tensor
        else:
            raise ValueError(f'Got {target} but expected NN, tomo or display')
    
