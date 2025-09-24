import os
import argparse
from tqdm import trange, tqdm
import SimpleITK as sitk
import numpy as np

def stitch_nii(root, subdir, steps, nr_slices=32, channels=4, height=128, width=128, verbose=False, force=False):
    if not force:
        stitched_path = os.path.join(root, "full_volumes", f"{subdir}_stitched_volume.nii")
        if os.path.exists(stitched_path):
            if verbose:
                print(f"Stitched file {stitched_path} already exists. Skipping stitching.")
            return None
    # make sure the output directory exists
    output_dir = os.path.join(root, "full_volumes")
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Processing: {subdir}")
    steps_dir = os.path.join(root, subdir, "time_steps", str(steps))
    if not os.path.exists(steps_dir):
        if verbose:
            print(f"Directory {steps_dir} does not exist. Skipping.")
        return None # Return None or raise an error to indicate failure

    # Read the first slice to get dimensions and data type for pre-allocation
    first_slice_path = os.path.join(steps_dir, f"{subdir}_slice_000.nii")
    if not os.path.exists(first_slice_path):
        if verbose:
            print(f"First slice {first_slice_path} not found. Cannot determine array shape. Skipping.")
        return None

    # Pre-allocate the array for the entire volume
    stitched_volume = np.zeros((nr_slices, channels, height, width), dtype=np.float32)  # Adjust dtype as needed

    if verbose:
        print(f"Pre-allocating volume of shape: {stitched_volume.shape}")

    for slice_idx in range(nr_slices):
        slice_path = os.path.join(steps_dir, f"{subdir}_slice_{slice_idx:03d}.nii")
        if not os.path.exists(slice_path):
            print(f"Warning: Slice {slice_path} does not exist. Filling with zeros for this slice.")
            stitched_volume[slice_idx] = 0 # Fill with zeros if a slice is missing
            continue
        try:
            slice_img = sitk.ReadImage(slice_path)
            stitched_volume[slice_idx] = sitk.GetArrayFromImage(slice_img)
        except Exception as e:
            print(f"Error reading slice {slice_path}: {e}. Filling with zeros for this slice.")
            stitched_volume[slice_idx] = 0 # Fill with zeros on error
    
    # modify the shape to match [D, H, W, C]
    stitched_volume = np.transpose(stitched_volume, (0, 2, 3, 1))  # Change to [D, H, W, C]
    
    if verbose:
        print(f"Stitched volume shape for {subdir}: {stitched_volume.shape}")
    
    return stitched_volume # Return the stitched NumPy array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch NIfTI files into a single 4D volume.")
    parser.add_argument("target_dir", help="Directory containing subdirectories with NIfTI files to stitch.")
    parser.add_argument("steps", type=int, help="Number of steps to stitch.")
    parser.add_argument("--nr_slices", type=int, default=32, help="Number of slices to expect in each time step. Default is 32.")
    parser.add_argument("--channels", type=int, default=4, help="Number of channels in the NIfTI files. Default is 4.")
    parser.add_argument("--height", type=int, default=128, help="Height of the slices. Default is 128.")
    parser.add_argument("--width", type=int, default=128, help="Width of the slices. Default is 128.")
        
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output, showing inner progress bars and detailed messages.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing stitched files.")
    
    args = parser.parse_args()
    target_dir = args.target_dir
    steps = args.steps
    nr_slices = args.nr_slices
    channels = args.channels
    height = args.height
    width = args.width
    
    verbose = args.verbose # Get the verbose flag
    force = args.force # Get the force flag

    # Ensure target_dir exists
    if not os.path.isdir(target_dir):
        print(f"Error: Target directory '{target_dir}' does not exist.")
        exit(1)

    dirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    
    # Only print these top-level messages if verbose is True, or if they are critical (like errors)
    if verbose:
        print(f"Found {len(dirs)} subdirectories in {target_dir}.")

    if not dirs:
        print(f"No subdirectories found in {target_dir}. Exiting.")
        exit(0)

    for d in tqdm(dirs, desc="Stitching directories"):
        # Pass the verbose flag to the stitch_nii function
        stitched_data = stitch_nii(target_dir, d, steps, nr_slices, channels, height, width, verbose=verbose, force=force)
        if stitched_data is not None:
            if verbose: # Only print this detailed success message if verbose
                print(f"Successfully stitched data for {d}. Shape: {stitched_data.shape}")
            # Save the stitched data as a NIfTI file
            stitched_path = os.path.join(target_dir, "full_volumes", f"{d}_stitched_volume_{steps}.nii")
            stitched_img = sitk.GetImageFromArray(stitched_data)
            sitk.WriteImage(stitched_img, stitched_path)
            if verbose:
                print(f"Stitched volume saved to {stitched_path}")
