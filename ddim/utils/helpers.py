import os
import SimpleITK as sitk
import numpy as np
import torch

## TODO: start from here... utils => to data load utils ...

def load_image_nii(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)

def load_patient_volumes(root_dir, patient_id, file_ext="nii"):
    """
    Loads ct and mr volumes for a patient from the root directory.
    The new file format is {patient_id}_latent_{ct|mr}.nii
    Returns: ct, mr, mask (all np.ndarrays). Mask is None as it's not in the new structure.
    """
    ct_path = os.path.join(root_dir, f"{patient_id}_latent_ct.{file_ext}")
    mr_path = os.path.join(root_dir, f"{patient_id}_latent_mr.{file_ext}")
    
    if file_ext == "nii":
        load_image = load_image_nii
        ct = load_image_nii(ct_path) if os.path.exists(ct_path) else None
        mr = load_image_nii(mr_path) if os.path.exists(mr_path) else None
        
    elif file_ext == "npy":
        ct = np.load(ct_path) if os.path.exists(ct_path) else None
        mr = np.load(mr_path) if os.path.exists(mr_path) else None
    
    # Mask is not available in the new data structure.
    return ct, mr, None

def get_patient_ids(root_dir, file_ext="nii"):
    """
    Yields unique patient IDs from file names in root_dir.
    """
    patient_ids = set()
    for filename in sorted(os.listdir(root_dir)):
        if f'_latent_ct.{file_ext}' in filename or f'_latent_mr.{file_ext}' in filename:
            patient_id = filename.split('_latent_')[0]
            patient_ids.add(patient_id)
    return list(patient_ids)
