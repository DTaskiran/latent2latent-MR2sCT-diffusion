import os
import csv
import shutil
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
import time
import math
import SimpleITK as sitk

from ddim.losses import hybrid_loss
from ddim.dataset import MRISCTDataset
from ddim.model import ConditionalUNetNew, ConditionalUNet2DViT
from ddim.config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    LOSS_TYPE,
    NUM_TRAIN_TIMESTEPS,
    DEVICE,
    TRAINING_NUM_WORKERS,
)
from ddim.utils.schedule_utils import setup_scheduler
from ddim.utils.loss_logger import LossLogger

def lr_cosine_decay(current_epoch, total_epochs, initial_lr, min_lr_factor=0.25):
    if current_epoch >= total_epochs:
        return initial_lr * min_lr_factor
    
    # Calculate the actual minimum learning rate
    lr_min = initial_lr * min_lr_factor
    
    # Calculate the range of learning rates to decay over
    lr_range = initial_lr - lr_min
    cosine_val = math.cos(math.pi * current_epoch / (total_epochs - 1))
    decayed_lr = lr_min + 0.5 * lr_range * (1 + cosine_val)
    
    return decayed_lr

def train():
    print(f"🧠 Using device: {DEVICE}")

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # CSV logger for per-slice loss
    loss_log_path = "logs/slice_losses.csv"
    with open(loss_log_path, "w") as f:
        csv.writer(f).writerow(["epoch", "batch_idx", "patient_idx", "slice_idx", "loss"])

    logger = LossLogger(run_name="unet_run")

    # Dataset
    dataset = MRISCTDataset(
        mode="train",
        use_preprocessed=True,
        num_workers=1,
    )

    if len(dataset) == 0:
        raise RuntimeError("❌ No training data found! Check your split file or preprocessing output.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=TRAINING_NUM_WORKERS,
        pin_memory=DEVICE.startswith("cuda"),
    )
    print(f"✅ Loaded {len(dataset)} training slices...")
    torch.set_float32_matmul_precision('high')
 
    model = ConditionalUNet2DViT().to(DEVICE)
    if DEVICE.startswith("cuda"):
        print('Compiling model...')
        model = torch.compile(model)
    use_fused = DEVICE.startswith('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, fused=use_fused)
    scheduler = setup_scheduler()
    best_loss = float('inf')
    model.train()
    scaler = torch.amp.GradScaler("cuda")
    
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    #autocast_dtype = torch.float32 # on 2080ti fix the autocast type, as bf16 does not seem to save memory or time
    print(f'Using autocast type {autocast_dtype}')
    
    # --- Start of Training Loop ---
    print("Time before first epoch:", time.strftime("%H:%M:%S", time.gmtime(time.time())))
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        total_loss = 0
        norm = 0

        # Apply cosine decay to the learning rate
        current_lr = lr_cosine_decay(epoch, EPOCHS, LR) # epoch - 1 because current_epoch is 0-indexed in the function
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            mri = batch["mri"].to(DEVICE)
            ct = batch["ct"].to(DEVICE)
            noise = torch.randn_like(ct, device=DEVICE) 
            
            B = ct.size(0)
            timesteps = torch.randint(0, NUM_TRAIN_TIMESTEPS, (B,), device=DEVICE)
            x_t = scheduler.add_noise(ct, noise, timesteps)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.split(":")[0], dtype=autocast_dtype):
                pred_noise = model(x_t, timesteps, mri).float()
                
            if (LOSS_TYPE == "simple"):
                batch_loss = F.mse_loss(pred_noise, noise, reduction="mean")
                # batch_loss = F.l1_loss(pred_noise.float(), noise)
            # # mse = F.mse_loss(pred_noise.float(), noise)
            # # l1 = F.l1_loss(pred_noise.float(), noise)
            # # batch_loss = mse * 0.5 + l1 * 0.5
            # grad_loss = gradient_loss(pred_noise, noise)
            # loss_l1 = F.l1_loss(pred_noise, noise)
            # loss_mse = F.mse_loss(pred_noise, noise)
            # batch_loss = 0.5 * loss_l1 + 0.25 * loss_mse + 0.25 * grad_loss
            
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            norm += torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += batch_loss.item() * B
            
            # with open(loss_log_path, "a") as f:
            #     writer = csv.writer(f)
            #     for i in range(B):
            #         writer.writerow([
            #             epoch,
            #             batch_idx,
            #             batch["patient_idx"][i].item(),
            #             batch["slice_idx"][i].item(),
            #             f"{batch_loss.item():.6f}",
            #         ])

            logger.log_batch(epoch, batch_idx, batch_loss.item())
            
        norm /= len(loader)
        # --- [FIX 3] Improved End of Epoch Summary ---
        logger.log_epoch(epoch, total_loss, len(dataset))
        total_epoch_time = time.time() - epoch_start_time
        loss_avg = total_loss / len(dataset)
        time_remaining_seconds = (EPOCHS - epoch) * total_epoch_time
        minutes_total, seconds = divmod(time_remaining_seconds, 60)
        hours, minutes = divmod(minutes_total, 60)

        print(f"Epoch {epoch:3}/{EPOCHS} | Avg. Loss: {loss_avg:.4f} | Time: {total_epoch_time:.2f} sec | Throughput: {B * len(loader) / total_epoch_time:.2f} slices/s | Time Remaining: {hours:2.0f}h {minutes:2.0f}m {seconds:2.0f}s | Norm {norm:.3f} | Current LR: {current_lr:.6f}")
        
        if epoch % 100 == 0:
            print(f'Storing checkpoint weights at epoch {epoch}')
            torch.save(model.state_dict(), "weights/conditional_unet_checkpoint.pth")
                
    # --- Save Model ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"weights/conditional_unet_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")

    config_source_path = "ddim/config.py"
    config_dest_path = f"weights/config_{timestamp}.py"

    try:
        shutil.copy(config_source_path, config_dest_path)
        print(f"📝 Config copied to: {config_dest_path}")
    except Exception as e:
        print(f"⚠️ Failed to copy config: {e}")
        
    logger.save_plot()


if __name__ == "__main__":
    train()
