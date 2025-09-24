#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK
import os
from tqdm import tqdm
import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter
import argparse

def load_nifti(path):
    image = SimpleITK.ReadImage(path)
    return SimpleITK.GetArrayFromImage(image)

class ImageMetrics():
    def __init__(self, debug=False):
        # Use fixed wide dynamic range
        self.dynamic_range = [-1024., 3000.]
        self.debug = debug
    def score_patient(self, gt_img, synthetic_ct, mask, patient_id=None):        
        assert gt_img.shape == synthetic_ct.shape, f'{gt_img.shape} != {synthetic_ct.shape} for patient {patient_id}'
        if mask is not None:
            assert mask.shape == synthetic_ct.shape 

        # perform masking on the images
        ground_truth = gt_img if mask is None else np.where(mask == 0, -1024, gt_img)
        prediction = synthetic_ct if mask is None else np.where(mask == 0, -1024, synthetic_ct)
        
        # Compute image similarity within the mask
        mae_value = self.mae(ground_truth,
                             prediction,
                             mask)
        
        psnr_value = self.psnr(ground_truth,
                               prediction,
                               mask,
                               use_population_range=True)
        
        ms_ssim_value, ms_ssim_mask_value = self.ms_ssim(ground_truth,
                               prediction, 
                               mask)

        return {
            'mae': mae_value,
            'psnr': psnr_value,
            'ms_ssim': ms_ssim_mask_value,
        }
    
    def mae(self,
            gt: np.ndarray, 
            pred: np.ndarray,
            mask: Optional[np.ndarray] = None) -> float:
        """
        Compute Mean Absolute Error (MAE)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
    
        Returns
        -------
        mae : float
            mean absolute error.
    
        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
        mae_value = np.sum(np.abs(gt*mask - pred*mask))/mask.sum() 
        return float(mae_value)
    
    
    def psnr(self,
             gt: np.ndarray, 
             pred: np.ndarray,
             mask: Optional[np.ndarray] = None,
             use_population_range: Optional[bool] = False) -> float:
        """
        Compute Peak Signal to Noise Ratio metric (PSNR)
    
        Parameters
        ----------
        gt : np.ndarray
            Ground truth
        pred : np.ndarray
            Prediction
        mask : np.ndarray, optional
            Mask for voxels to include. The default is None (including all voxels).
        use_population_range : bool, optional
            When a predefined population wide dynamic range should be used.
            gt and pred will also be clipped to these values.
    
        Returns
        -------
        psnr : float
            Peak signal to noise ratio..
    
        """
        if mask is None:
            mask = np.ones(gt.shape)
        else:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
        if use_population_range:            
            # Clip gt and pred to the dynamic range
            gt = np.clip(gt, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            pred = np.clip(pred, a_min=self.dynamic_range[0], a_max=self.dynamic_range[1])
            dynamic_range = self.dynamic_range[1]  - self.dynamic_range[0]
        else:
            dynamic_range = gt.max()-gt.min()
            pred = np.clip(pred, a_min=gt.min(), a_max=gt.max())
            
        # apply mask
        gt = gt[mask==1]
        pred = pred[mask==1]
        psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
        return float(psnr_value)

    # Compute the luminance, contrast and structure components of the SSIM between two images    
    def structural_similarity_at_scale(self, im1,
        im2,
        *,
        luminance_weight=1,
        contrast_weight=1,
        structure_weight=1,
        win_size=None,
        gradient=False,
        data_range=None,
        channel_axis=None,
        gaussian_weights=False,
        full=False,
        **kwargs,):
            K1 = kwargs.pop('K1', 0.01)
            K2 = kwargs.pop('K2', 0.03)
            sigma = kwargs.pop('sigma', 1.5)
            if K1 < 0:
                raise ValueError("K1 must be positive")
            if K2 < 0:
                raise ValueError("K2 must be positive")
            if sigma < 0:
                raise ValueError("sigma must be positive")
            use_sample_covariance = kwargs.pop('use_sample_covariance', True)
            if gaussian_weights:
                # Set to give an 11-tap filter with the default sigma of 1.5 to match
                # Wang et. al. 2004.
                truncate = 3.5

            if win_size is None:
                if gaussian_weights:
                    # set win_size used by crop to match the filter size
                    r = int(truncate * sigma + 0.5)  # radius as in ndimage
                    win_size = 2 * r + 1
                else:
                    win_size = 7  # backwards compatibility
            if gaussian_weights:
                filter_func = gaussian
                filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
            else:
                filter_func = uniform_filter
                filter_args = {'size': win_size}

            ndim = im1.ndim
            NP = win_size**ndim

            # filter has already normalized by NP
            if use_sample_covariance:
                cov_norm = NP / (NP - 1)  # sample covariance
            else:
                cov_norm = 1.0  # population covariance to match Wang et. al. 2004
            # compute (weighted) means
            ux = filter_func(im1, **filter_args)
            uy = filter_func(im2, **filter_args)


            # compute (weighted) variances and covariances
            uxx = filter_func(im1 * im1, **filter_args)
            uyy = filter_func(im2 * im2, **filter_args)
            uxy = filter_func(im1 * im2, **filter_args)
            vx = cov_norm * (uxx - ux * ux)
            vxsqrt = np.clip(vx, a_min=0, a_max=None) ** 0.5 # TODO: this is very ugly
            vy = cov_norm * (uyy - uy * uy)
            vysqrt = np.clip(vy, a_min=0, a_max=None) ** 0.5 # TODO: this is very ugly
            vxy = cov_norm * (uxy - ux * uy)

            R = data_range
            C1 = (K1 * R) ** 2
            C2 = (K2 * R) ** 2
            C3 = C2 / 2

            L = np.clip((2 * ux * uy + C1) / (ux * ux + uy * uy + C1), a_min=0, a_max=None) # TODO is this clipping necessary or do we increase K1 and K2?

            C = np.clip((2 * vxsqrt * vysqrt + C2) / (vx + vy + C2), a_min=0, a_max=None)
            S = np.clip((vxy + C3) / (vxsqrt * vysqrt + C3), a_min=0, a_max=None)

            result = (L ** luminance_weight) * (C ** contrast_weight) * (S ** structure_weight)
            # to avoid edge effects will ignore filter radius strip around edges
            pad = (win_size - 1) // 2

            # compute (weighted) mean of ssim. Use float64 for accuracy.
            mssim = crop(result, pad).mean(dtype=np.float64)

            if full:
                return mssim, result
            return mssim


    # Compute the masked MS-SSIM by masking the SSIM at every resolution level
    def ms_ssim(self, gt: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray] = None, scale_weights: Optional[np.ndarray] = None) -> float:
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(self.dynamic_range), max(self.dynamic_range))
        pred = np.clip(pred, min(self.dynamic_range), max(self.dynamic_range))

        if mask is not None:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
            # Mask gt and pred
            gt = np.where(mask==0, min(self.dynamic_range), gt)
            pred = np.where(mask==0, min(self.dynamic_range), pred)

        # Make values non-negative
        if min(self.dynamic_range) < 0:
            gt = gt - min(self.dynamic_range)
            pred = pred - min(self.dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = self.dynamic_range[1] - self.dynamic_range[0]


        # see Eq. 7 in https://live.ece.utexas.edu/publications/2003/zw_asil2003_msssim.pdf
        # Also, the final sentence of section 3.2 (Results)
        scale_weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]) if scale_weights is None else scale_weights
        luminance_weights = np.array([0, 0, 0, 0, 0, 0.1333]) if scale_weights is None else scale_weights 
        levels = len(scale_weights)

        downsample_filter = np.ones((2, 2, 2)) / 8

        gtx, gty, gtz = gt.shape

        # Due to the downsampling in the MS-SSIM, the minimum matrix size must be 97 in every dimension
        target_size = 97

        pad_values = [
            (np.clip((target_size - dim)//2, a_min=0, a_max=None), 
            np.clip(target_size - dim - (target_size - dim)//2, a_min=0, a_max=None)) 
            for dim in [gtx, gty, gtz]]

        gt   = np.pad(gt,   pad_values, mode='edge')
        pred = np.pad(pred, pad_values, mode='edge')
        mask = np.pad(mask, pad_values, mode='edge')

        min_size = (downsample_filter.shape[-1] - 1) * 2 ** (levels - 1) + 1

        ms_ssim_vals, ms_ssim_maps = [], []
        for level in range(levels):
            ssim_value_full, ssim_map = self.structural_similarity_at_scale(gt, pred, 
                luminance_weight=luminance_weights[level], 
                contrast_weight=scale_weights[level],
                structure_weight=scale_weights[level],
                data_range=dynamic_range, full=True)
            pad = 3
            # at every level, we get the ssim_value_full, which is just mean SSIM at a level L, and the 
            # SSIM map. The masked SSIM is the mean SSIM within this mask
            ssim_value_masked  = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)

            ms_ssim_vals.append(ssim_value_full)
            ms_ssim_maps.append(ssim_value_masked)

            # The images are cleverly downsampled using an uniform filter
            # the mask is just downsampled by selecting every other line in every dimension
            filtered = [fftconvolve(im, downsample_filter, mode='same')
                for im in [gt, pred]]
            gt, pred, mask = [x[::2, ::2, ::2] for x in [*filtered, mask]]

        ms_ssim_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_vals])
        ms_ssim_mask_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_maps])

        return float(ms_ssim_val), float(ms_ssim_mask_val)


if __name__=='__main__':
    # get path to ground truth, prediction and mask directories from command line arguments
    parser = argparse.ArgumentParser(description="Calculate image metrics for MRI/CT images.")
    parser.add_argument('--predicted', type=str, required=True, help="Path to directory containing predicted images.")
    parser.add_argument('--raws_dir', type=str, required=True, help="Path to directory containing raw images.")
    parser.add_argument('--quiet', action='store_true', help="If set, will not print per-patient results.")
    args = parser.parse_args()
    predicted_path = args.predicted
    raws_dir = args.raws_dir

    metrics = ImageMetrics()
    results = {}
    meta_info = {}  # store body_part and center for grouping

    for synth_scan in tqdm(os.listdir(predicted_path)):
        patient_id = synth_scan.split('_')[0]
        body_part = synth_scan.split("_")[0][1:3]  # TH, HN, AB
        center = patient_id[3]  # A, B, C or D

        if not center == "D":
            mask_root_path = os.path.join(raws_dir, 'synthRAD2025_Task1_Train/Task1/')
        else:
            mask_root_path = os.path.join(raws_dir, 'synthRAD2025_Task1_Train_D/Task1/')
        mask_file = os.path.join(mask_root_path, body_part, patient_id, "mask.mha")
        if not os.path.exists(mask_file):
            print(f"Mask file not found for {patient_id}, skipping this case.")
            continue

        synth_scan_path = os.path.join(predicted_path, synth_scan)
        ct_path = os.path.join(mask_root_path, body_part, patient_id, "ct.mha")

        ct_img = load_nifti(ct_path)
        synth_img = load_nifti(synth_scan_path)
        mask = load_nifti(mask_file)
        try:
            scores = metrics.score_patient(ct_img, synth_img, mask, patient_id=patient_id)
            results[patient_id] = scores
            meta_info[patient_id] = (body_part, center)
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            continue

    # Print per-patient results
    for patient_id, scores in results.items():
        if args.quiet:
            break
        print(f"Patient {patient_id}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")

    # Overall metrics
    all_scores = np.array([list(scores.values()) for scores in results.values()])
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)

    print(f"\nOverall Metrics (Nr. of patients={len(all_scores)}):")
    for i, metric in enumerate(next(iter(results.values())).keys()):
        print(f"{metric}: mean={mean_scores[i]:.4f}, std={std_scores[i]:.4f}")

    # Per body part metrics
    print("\nMetrics per body part:")
    body_parts = sorted(set(bp for bp, _ in meta_info.values()))
    for bp in body_parts:
        bp_scores = np.array([list(results[pid].values()) for pid, (part, _) in meta_info.items() if part == bp])
        if bp_scores.size == 0:
            continue
        bp_mean = np.mean(bp_scores, axis=0)
        bp_std = np.std(bp_scores, axis=0)
        print(f"  Body part {bp} (n={len(bp_scores)}):")
        for i, metric in enumerate(next(iter(results.values())).keys()):
            print(f"    {metric}: mean={bp_mean[i]:.4f}, std={bp_std[i]:.4f}")

    # Per center metrics
    print("\nMetrics per center:")
    centers = sorted(set(c for _, c in meta_info.values()))
    for c in centers:
        c_scores = np.array([list(results[pid].values()) for pid, (_, center) in meta_info.items() if center == c])
        if c_scores.size == 0:
            continue
        c_mean = np.mean(c_scores, axis=0)
        c_std = np.std(c_scores, axis=0)
        print(f"  Center {c} (n={len(c_scores)}):")
        for i, metric in enumerate(next(iter(results.values())).keys()):
            print(f"    {metric}: mean={c_mean[i]:.4f}, std={c_std[i]:.4f}")
