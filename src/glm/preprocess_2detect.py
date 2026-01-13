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
        mode='mode2'
        ):
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
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython preprocess_2detect.py mode task \n")
        sys.exit(1)

    parameters = yaml.safe_load(open("params.yaml"))

    mode = sys.argv[1]
    accepted_modes = ['mode1', 'mode2', 'mode3']
    assert mode in accepted_modes, f"Wrong mode argument, expected to be in {accepted_modes}, got {mode}"

    task = sys.argv[2]
    accepted_tasks = ['demo', 'train']
    assert task in accepted_tasks, f"Wrong task argument, expected to be in {accepted_tasks}, got {mode}"

    raw_path = pathlib.Path(parameters[task]['data']['raw_path'])
    processed_path = pathlib.Path(parameters[task]['data']['processed_path'])
    loop_over_dataset(raw_path, processed_path, mode)

if __name__ == '__main__':
    main()