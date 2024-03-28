import matplotlib.pyplot as plt
import matplotlib
import torch
from models.ema import ExponentialMovingAverage

from pathlib import Path
import controllable_generation
from utils import restore_checkpoint, clear_color, clear

import models
from models import utils as mutils
from models import ncsnpp
import sampling
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)

import datasets
from skimage.transform import resize
from torchvision import transforms

from utils import normalize_np
from utils import get_logger
from models.condition_methods import get_condition_method
from models.measurements import get_operator,get_noise
import importlib
import numpy as np

def main():
    
    root = 'samples'
    num_scales = 2000
    sde = 'VESDE'
    
    print('initaializing...')
    if sde.lower() == 'vesde':
        configs = importlib.import_module(f"configs.ve.fastmri_knee_720_ncsnpp_continuous")
        config = configs.get_config()
        config.model.num_scales = num_scales
        ckpt_filename = '/home/arpanp/Downloads/score-MRI/work_dir/checkpoints/checkpoint_980.pth'
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    
    batch_size = 1
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    
    # logger
    device=config.device
    logger = get_logger()
    logger.info(f"Device set to {device}.")
    
    #Prepare Operator and noise
    measure_config = config.measurement
    operator=get_operator(device=device, **measure_config.operator)
    noiser = get_noise(**measure_config.noise)
    logger.info(f"Operation: {measure_config.operator.name} / Noise: {measure_config.noise.name}")
    
    # Prepare conditioning method
    cond_config = config.conditioning
    cond_method = get_condition_method(cond_config.method, operator, noiser, **cond_config.params)
    #print(cond_method)
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {cond_config.method}")
    
    

    random_seed = 0
    
    #score model
    
    sigmas = mutils.get_sigmas(config)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    ema = ExponentialMovingAverage(score_model.parameters(),
                                decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device, skip_optimizer=True)
    ema.copy_to(score_model.parameters())

    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    snr = 0.16
    n_steps = 1
    probability_flow = False
    
    idx=6
    # for idx in range(9):
    filename = Path(root) /'lr'/ (str(idx).zfill(6) + '.npy')
    # Specify save directory for saving generated samples
    save_root = Path(f'./results_final/{idx}')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    # Read data
    img = torch.from_numpy(normalize_np(np.load(filename)))
    h, w = img.shape
    img = img.view(1, 1, h, w)
    img = img.to(config.device)
    #label
    l_filename = Path(root) /'hr'/ (str(idx).zfill(6) + '.npy')
    label = torch.from_numpy(normalize_np(np.load(l_filename)))
    lh, lw = label.shape
    label = label.view(1, 1, lh, lw)
    label = label.to(config.device)
    plt.imsave(save_root / 'input' / f'{str(idx).zfill(6)}.png', clear(img), cmap='gray')
    plt.imsave(save_root / 'label' / f'{str(idx).zfill(6)}.png', clear(label), cmap='gray')
    ###############################################
    #Inference
    ###############################################
    
    pc_mri=controllable_generation.get_pc_mri(sde,
                                            predictor, corrector,
                                            inverse_scaler,
                                            snr=snr,
                                            n_steps=n_steps,
                                            probability_flow=probability_flow,
                                            continuous=config.training.continuous,
                                            denoise=True,
                                            save_progress=True,
                                            save_root=save_root,measurement_cond_fn=measurement_cond_fn)

    x = pc_mri(score_model,scaler(img),label.shape)
    
    # random_sampler=sampling.get_pc_sampler(sde=sde,shape=label.shape,
    #                                          predictor=predictor,corrector= corrector,
    #                                          inverse_scaler=inverse_scaler,
    #                                          snr=snr,
    #                                          n_steps=n_steps,
    #                                          probability_flow=probability_flow,
    #                                          continuous=config.training.continuous,
    #                                          denoise=True)
    # x,_=random_sampler(score_model)
    # Recon
    np.save(save_root / 'recon' / f'{str(idx).zfill(5)}.npy', clear(x))
    plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x), cmap='gray')
    plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}_clip.png'), np.clip(clear(x), 0.05, 0.95), cmap='gray')
        
    
    


if __name__ == '__main__':
    main()
