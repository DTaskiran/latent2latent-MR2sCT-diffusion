import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict
from tqdm import tqdm
from multiprocessing import Pool

# Assuming these are defined elsewhere as in the original code.
# You may need to adjust the import paths based on your project structure.
from ddim.config import (
    MODEL_DIM,
    ORIGINAL_DATA_ROOT,
    PREPROCESSED_DATA_ROOT,
    SPLITS_DIR,
    TRAINING_NUM_WORKERS,
    INFERENCE_NUM_WORKERS,
)
from ddim.preprocess import process_patient
from ddim.utils.helpers import get_patient_ids, load_patient_volumes

class MRISCTDataset(Dataset):
    """
    MRI-SCT Dataset (sequential-only).

    - Loads all 2D slices into memory on init.
    - Respects train/val splits and optional per-slice exclusions.
    """

    def __init__(
        self,
        mode: str = "train",
        transform: Optional[Callable] = None,
        use_preprocessed: bool = True,
        allowed_patients: Optional[list] = None,
    ):
        assert mode in ["train", "val", "infer"], "Invalid mode"
        if MODEL_DIM != "2D":
            raise NotImplementedError("This Dataset only supports MODEL_DIM='2D'.")

        self.mode = mode
        self.transform = transform
        self.use_preprocessed = use_preprocessed
        self.root = PREPROCESSED_DATA_ROOT if use_preprocessed else ORIGINAL_DATA_ROOT

        # Exclusions (train only)
        self.exclusion_dict = {}
        self.slice_exclusion_log = []
        if self.mode == "train":
            exclusion_path = os.path.join(SPLITS_DIR, "mr.mha-comment_original.json")
            if os.path.exists(exclusion_path):
                with open(exclusion_path, "r") as f:
                    self.exclusion_dict = json.load(f)
            else:
                print(f"⚠️ Exclusion file not found at: {exclusion_path}. No slices will be excluded.")

        # Patients to load
        if mode in ["train", "val"]:
            split_file = os.path.join(SPLITS_DIR, f"{mode}.txt")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Missing split file: {split_file}")
            with open(split_file) as f:
                self.allowed_patients = set(line.strip() for line in f if line.strip())
        else:
            self.allowed_patients = set(allowed_patients) if allowed_patients else None

        # Storage
        self.slices = []
        self.patient_ids_loaded = []

        # Load
        self._load_and_cache_slices_sequential()

        # Map patient_id -> integer index
        if self.slices:
            self.patient_ids_loaded = sorted({s["patient_id_str"] for s in self.slices})
            pid_to_idx = {pid: i for i, pid in enumerate(self.patient_ids_loaded)}
            for s in self.slices:
                s["patient_idx"] = pid_to_idx.pop(s.pop("patient_id_str"), None)

        # Logs
        print(f"\n📂 [{self.mode}] Loaded data for {len(self.patient_ids_loaded)} patients from {self.root}")
        if self.mode in ["train", "val"]:
            print(f"🗞️ [{self.mode}] Patient IDs used: {sorted(self.patient_ids_loaded)}")
            os.makedirs("logs", exist_ok=True)
            with open(f"logs/used_patients_{self.mode}.txt", "w") as f:
                for pid in sorted(self.patient_ids_loaded):
                    f.write(f"{pid}\n")
            if self.mode == "train" and self.slice_exclusion_log:
                with open("logs/excluded_slices_train.txt", "w") as f:
                    for line in self.slice_exclusion_log:
                        f.write(f"{line}\n")

        print(f"👁️ [{self.mode}] Total usable slices pre-loaded into memory: {len(self.slices)}")

    def _eligible_patients(self):
        file_ext = "npy" if self.use_preprocessed else "nii"
        all_ids = get_patient_ids(self.root, file_ext=file_ext)
        eligible = []
        for pid in all_ids:
            if self.allowed_patients is not None and pid not in self.allowed_patients:
                continue
            if self.mode == "train":
                info = self.exclusion_dict.get(pid, {})
                if not info.get("include", True):
                    self.slice_exclusion_log.append(f"🚫 Skipping entire patient (marked for exclusion): {pid}")
                    continue
            eligible.append(pid)
        return eligible

    def _load_and_cache_slices_sequential(self):
        file_ext = "npy" if self.use_preprocessed else "nii"
        print(f"🔍 Loading and caching all slices from: {self.root} (use_preprocessed={self.use_preprocessed})")

        for patient_id in tqdm(self._eligible_patients(),
                               desc="Loading and Caching Slices", leave=False):
            try:
                # If raw data is requested, ensure preprocessing exists.
                if not self.use_preprocessed:
                    process_patient(patient_id, self.root)

                ct_vol, mr_vol, _ = load_patient_volumes(self.root, patient_id, file_ext=file_ext)
                if mr_vol is None or ct_vol is None:
                    print(f"⚠️ Skipping patient {patient_id} due to missing MR or CT volume.")
                    continue

                # Ensure channel-last
                if mr_vol.ndim == 3:
                    mr_vol = np.expand_dims(mr_vol, axis=-1)
                if ct_vol.ndim == 3:
                    ct_vol = np.expand_dims(ct_vol, axis=-1)

                mr_vol_t = torch.from_numpy(mr_vol).permute(3, 0, 1, 2).float()  # (C, Z, H, W)
                ct_vol_t = torch.from_numpy(ct_vol).permute(3, 0, 1, 2).float()

                num_slices = mr_vol_t.shape[1]
                excluded_slices = set()
                if self.mode == "train" and patient_id in self.exclusion_dict:
                    excluded_slices = set(self.exclusion_dict[patient_id].get("excluded_slices", []))

                for s_idx in range(num_slices):
                    if s_idx in excluded_slices:
                        self.slice_exclusion_log.append(f"❌ Skipping slice {s_idx} for patient {patient_id}")
                        continue

                    sample: Dict[str, torch.Tensor] = {
                        "mri": mr_vol_t[:, s_idx, :, :],
                        "patient_id_str": patient_id,
                        "slice_idx": s_idx,
                    }
                    if self.mode in ["train", "val", "infer"]:
                        sample["ct"] = ct_vol_t[:, s_idx, :, :]

                    self.slices.append(sample)

            except Exception as e:
                print(f"⚠️ Error processing patient {patient_id}, skipping. Error: {e}")

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.slices[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
