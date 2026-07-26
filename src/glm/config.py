from pathlib import Path
from typing import Dict

import yaml
import torch
from ignite.engine import Engine
from dvclive import Live

PRETRAINED_SINOGRAM_MODEL_PATH = Path('src/glm/saved_models/pretrained_sinogram_model.pt')
COMBINED_MODEL_PATH = Path('src/glm/saved_models/model.pt')

def load_params(path: str = "params.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)

def init_run(seed: int) -> torch.device:
    torch.manual_seed(seed)
    return torch.device('cuda:0')

def log_ignite_metrics(evaluator: Engine, test_dataloader, live: Live, downsampling: int) -> Dict[str, float]:
    state = evaluator.run(test_dataloader)

    test_psnr = state.metrics['psnr']
    test_ssim = state.metrics['ssim']
    print(f"Computed SSIM: {test_ssim:.4f}")
    print(f"Computed PSNR: {test_psnr:.4f}")

    live.log_metric("Test SSIM", test_ssim)
    live.log_metric("Test PSNR", test_psnr)
    live.log_param('Downsampling', downsampling)

    return {'psnr': test_psnr, 'ssim': test_ssim}
