import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from ddim.config import ORIGINAL_DATA_ROOT, PREPROCESS_NUM_WORKERS, PREPROCESSED_DATA_ROOT, TARGET_SIZE
from ddim.utils.helpers import get_patient_ids, load_patient_volumes

#========================================================================================
# unused helper functions for normalization, made redundant by simple clamp+divide preprocessing
# left here for reference/future use if needed
def normalize_01(t: torch.Tensor) -> torch.Tensor:
  """Normalizes a tensor to the [0, 1] range."""
  return (t - t.min()) / (t.max() - t.min())

def normalize_11(t: torch.Tensor) -> torch.Tensor:
    return normalize_01(t) * 2 - 1

def quantile_filer(x: torch.Tensor, min_q=0.01, max_q=0.99):
    lower_bound = torch.quantile(x, min_q)
    upper_bound = torch.quantile(x, max_q)
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_masked = x[mask]
    assert x_masked.numel() == x.numel(), "Quantile filtering did not preserve the number of elements"
    return x_masked

def abs_filter(x: torch.Tensor, min_val=-2.2, max_val=2.2):
    mask = (x >= min_val) & (x <= max_val)
    x_masked = x[mask]
    assert x_masked.numel() == x.numel(), "Absolute value filtering did not preserve the number of elements"
    return x_masked

def normalize_to_range(tensor: torch.Tensor, range_min: float, range_max: float) -> torch.Tensor:
  # Ensure the range values are floats to guarantee float division
  range_min = float(range_min)
  range_max = float(range_max)

  # Calculate the span of the custom input range
  span = range_max - range_min
  normalized_tensor = (tensor - range_min) / span
  return normalized_tensor
#========================================================================================

def process_patient(patient_id, root_dir):
    pid = patient_id
    try:
        ct, mr, _ = load_patient_volumes(root_dir, pid)
        ct = np.zeros_like(mr)
        if ct is None or mr is None:
            return (pid, False, "Missing CT or MR file")
        #print(f"Patient volume loaded {pid}")

        ## Preprocess
        ct_processed = torch.from_numpy(ct)
        mr_processed = torch.from_numpy(mr)
        
        # apply clamping to [-2, 2] to remove outlies
        # value chosen based on +-3 stddev from mean of latent values in full latent dataset (SynthRAD2025, encoded with MAISI VAE)
        ct_processed = torch.clamp(ct_processed, -2, 2)
        mr_processed = torch.clamp(mr_processed, -2, 2)

        # #normalizing tensors to [-1, 1]
        ct_processed /= 2 #max abs value is 2
        mr_processed /= 2 #max abs value is 2
        
        #Save preprocessed 
        output_path = os.path.join(PREPROCESSED_DATA_ROOT)
        os.makedirs(output_path, exist_ok=True)
        np.save(os.path.join(output_path, f"{pid}_latent_mr.npy"), mr_processed.to(torch.float32).numpy())
        np.save(os.path.join(output_path, f"{pid}_latent_ct.npy"), ct_processed.to(torch.float32).numpy())
        
        #print(f"Proprocessed saved! {pid}")
        return (pid, True, f"Processed {len(mr_processed)} slices, saved to {output_path}")

    except Exception as e:
        print(f"Error processing patient {pid}: {e}")
        return (pid, False, str(e))


def main():
    os.makedirs(PREPROCESSED_DATA_ROOT, exist_ok=True)
    patient_ids = get_patient_ids(ORIGINAL_DATA_ROOT)
    print(f"⚙️ Launching preprocessing on {len(patient_ids)} patients using {PREPROCESS_NUM_WORKERS} workers...")

    process_func = partial(process_patient, root_dir=ORIGINAL_DATA_ROOT)
    with Pool(processes=PREPROCESS_NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_func, patient_ids), total=len(patient_ids)))

    saved = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print("\n✅ Preprocessing complete.")
    print(f"✅ Saved: {len(saved)} patients to: {PREPROCESSED_DATA_ROOT}")

    if failed:
        print(f"\n⚠️ Skipped: {len(failed)} patients")
        for pid, _, reason in failed:
            print(f"   - {pid}: {reason}")


if __name__ == "__main__":
    main()
