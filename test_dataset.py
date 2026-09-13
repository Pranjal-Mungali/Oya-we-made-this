"""
Test script for Sentinel-2 Super-Resolution Dataset Pipeline.
"""

from src.dataset import Sentinel2SRDataset


def main():
    print("Testing Sentinel2SRDataset...")

    # 1. Create dataset (4 bands: RGB + NIR, 2x scale, HR patch size 128x128)
    dataset = Sentinel2SRDataset(
        scale_factor=2,
        hr_patch_size=128,
        channels=4,
        num_samples=10,
        is_train=True
    )

    # 2. Load one sample
    lr_sample, hr_sample = dataset[0]

    # 3. Print shapes of LR and HR
    print(f"LR Patch Shape: {list(lr_sample.shape)} (Channels, Height, Width)")
    print(f"HR Patch Shape: {list(hr_sample.shape)} (Channels, Height, Width)")

    # 4. Confirm that LR is lower resolution than HR
    assert lr_sample.shape[-1] < hr_sample.shape[-1] and lr_sample.shape[-2] < hr_sample.shape[-2], (
        "LR must have lower spatial resolution than HR"
    )
    print("Confirmed: LR is lower resolution than HR (64x64 vs 128x128).")


if __name__ == "__main__":
    main()
