import pathlib 
import sys

import yaml
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm

from odl.contrib.datasets.ct.detect import preprocess_sinogram

# Folder Organisation 
# datasets
#     - raw
#         - 2detect
#             - slice00001
#                 - Mode1
#                     - dark.tif
#                     - ...

def loop_over_dataset(
        raw_path:pathlib.Path, 
        processed_path:pathlib.Path, 
        mode:str
        ):
    print(f'Processing the 2detect dataset: \n')
    print(f'\t Raw filepath: {raw_path}\n')
    print(f'\t Processed filepath: {processed_path}\n')

    assert raw_path.glob(f'*/{mode}') is not [], f'There are no files matching the 2detect pattern 2detect/slice0000x/{mode}, aborting.'

    for path_to_slice in tqdm(raw_path.glob(f'*/{mode}')):
        slice_name = path_to_slice.parent.stem
        path_to_processed_folder = processed_path.joinpath(f'{slice_name}/{mode}')
        path_to_processed_folder.mkdir(exist_ok=True, parents=True)
        # Process the sinogram
        path_to_processed = path_to_processed_folder.joinpath('preprocessed_sinogram.npy')
        if not path_to_processed.is_file():
            sinogram = preprocess_sinogram(path_to_slice)
            np.save(path_to_processed, sinogram)
        # Process the reconstruction
        path_to_processed = path_to_processed_folder.joinpath('preprocessed_reconstruction.npy')
        if not path_to_processed.is_file():
            reconstruction = np.asarray(imageio.imread(path_to_slice.joinpath("reconstruction.tif")))
            np.save(path_to_processed, reconstruction)

def main():
    parameters = yaml.safe_load(open("params.yaml"))

   
    raw_path = pathlib.Path(parameters['data']['raw_path'])
    processed_path = pathlib.Path(parameters['data']['processed_path'])

    assert raw_path.is_dir(), f'The input raw_path {raw_path} is not a dir.'

    for mode in ['mode1', 'mode2', 'mode3']:
        loop_over_dataset(raw_path, processed_path, mode)

if __name__ == '__main__':
    main()