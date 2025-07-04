# Skin-Based-Hand-Segmentation-Using-RGB-and-HSV-Thresholding
Implemented hand segmentation using RGB and HSV color thresholding. Analyzed color channel histograms to detect skin pixels and apply binary masks. Evaluated segmentation results to determine the more effective color space for skin detection.

# Hand Segmentation Using Color Thresholding

## Overview
This project implements the first stage of an Automatic Sign Language Recognition System (ASLRS) by segmenting hand regions from images using color-based thresholding. The segmentation is applied in both RGB and HSV color spaces.

## Features
- Reads a dataset of hand images from "HandDataset/"
- Computes and plots histograms for R, G, B and H, S, V channels
- Applies manually chosen thresholds to segment hand regions
- Outputs binary images where hands are highlighted (foreground in black, background in white)
- Compares results between RGB and HSV segmentation

## How It Works
1. **Image Reading:** Loads hand images as RGB from the dataset.
2. **Histogram Analysis:** Computes and displays 3-channel histograms to analyze pixel intensity distribution.
3. **Thresholding:** Applies hardcoded thresholds:
   - RGB: (150–255) for all channels
   - HSV: (H: 0–120), (S: 0–30), (V: 150–190)
4. **Segmentation:** Creates binary masks and outputs segmented images.
5. **Saving Results:** Saves output images in folders `RGBSegmentedDataset/` and `HSVSegmentedDataset/`.

## Requirements
- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Run the Code
Place the hand images in a folder named `HandDataset/` (excluding image `2_rendered.png` if missing), then run:

```bash
python main.py
