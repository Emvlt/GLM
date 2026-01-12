import pathlib 
import sys

import numpy as np
import imageio.v2 as imageio

from odl.contrib.datasets.ct.detect import preprocess_sinogram

# Folder Organisation 
# datasets
#     - raw
#         - 2detect
#             - slice00001
#                 - Mode1
#                     - dark.tif
#                     - ...

RAW_PATH = pathlib.Path('datasets/raw/2detect')
PROCESSED_PATH = pathlib.Path('datasets/processed/2detect')

def loop_over_dataset(mode):
    for path_to_slice in RAW_PATH.glob(f'*/{mode}'):
        slice_name = path_to_slice.parent.stem
        path_to_processed_folder = PROCESSED_PATH.joinpath(f'{slice_name}/{mode}')
        path_to_processed_folder.mkdir(exist_ok=True, parents=True)
        # Process the sinogram
        path_to_processed = path_to_processed_folder.joinpath('sinogram.npy')
        if not path_to_processed.is_file():
            sinogram = preprocess_sinogram(path_to_slice)
            np.save(path_to_processed, sinogram)
        # Process the reconstruction
        path_to_processed = path_to_processed_folder.joinpath('reconstruction.npy')
        if not path_to_processed.is_file():
            reconstruction = np.asarray(imageio.imread(path_to_slice.joinpath("reconstruction.tif")))
            np.save(PROCESSED_PATH.joinpath(f'{slice_name}/{mode}/reconstruction.npy'), reconstruction)

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython preprocess_2detect.py mode \n")
        sys.exit(1)

    mode = sys.argv[1]
    assert mode in ['Mode1', 'Mode2', 'Mode3'], f"Wrong mode argument, expected to be in ['Mode1', 'Mode2', 'Mode3'], got {mode}"

    loop_over_dataset(mode)

if __name__ == '__main__':
    main()