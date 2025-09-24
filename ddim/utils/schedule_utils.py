import torch
import math
from diffusers import DDIMScheduler

from ddim.config import DEVICE, NUM_INFER_STEPS, NUM_TRAIN_TIMESTEPS

def cosine_beta_schedule(num_steps: int, s: float = 0.008): ## TODO: Check impl & alternatives
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps) / num_steps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clip(1e-4, 0.999)

def setup_scheduler(num_steps: int = NUM_TRAIN_TIMESTEPS, device=DEVICE, num_infer_steps=NUM_INFER_STEPS, eta: float = 1):
    betas = cosine_beta_schedule(num_steps) #.to(device)

    scheduler = DDIMScheduler(num_train_timesteps=num_steps, clip_sample=False)
    scheduler.eta = eta  # Set eta directly as attribute

    if num_infer_steps:
        scheduler.set_timesteps(num_infer_steps)

    scheduler.betas = betas
    scheduler.alphas = 1.0 - betas
    scheduler.alphas_cumprod = torch.cumprod(scheduler.alphas, dim=0)

    return scheduler
