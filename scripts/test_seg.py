#!/usr/bin/env python
"""
Test script for VxmDenseSemiSupervisedSeg models.

Usage:
    python test_seg.py \
        --model ./models_seg_scan/0000.h5 \
        --pairs ./neurite-oasis.2d.v1.0/pairs.txt \
        --img-suffix slice_norm.nii.gz \
        --seg-suffix slice_seg4.nii.gz \
        --nb-labels 4
"""

import os
import argparse
import time
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
from voxelmorph.networks import VxmDenseSemiSupervisedSeg

# -------------------------
# Parse commandline args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=None, help='GPU number')
parser.add_argument('--model', required=True, help='trained VxmDenseSemiSupervisedSeg weights (.h5)')
parser.add_argument('--pairs', required=True, help='list of image pairs (txt file)')
parser.add_argument('--img-suffix', help='suffix for image files')
parser.add_argument('--seg-suffix', help='suffix for segmentation files')
parser.add_argument('--img-prefix', default='', help='optional image prefix')
parser.add_argument('--seg-prefix', default='', help='optional segmentation prefix')
parser.add_argument('--nb-labels', type=int, required=True, help='number of segmentation labels')
parser.add_argument('--multichannel', action='store_true', help='if data has multiple channels')
args = parser.parse_args()

# -------------------------
# Device setup
# -------------------------
device, nb_devices = vxm.utils.setup_device(args.gpu)
add_feat_axis = not args.multichannel

# -------------------------
# Read pairs
# -------------------------
img_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.img_prefix, suffix=args.img_suffix)
seg_pairs = vxm.py.utils.read_pair_list(args.pairs, prefix=args.seg_prefix, suffix=args.seg_suffix)

# -------------------------
# Load a sample to get input shape
# -------------------------
sample_img = vxm.py.utils.load_volfile(img_pairs[0][0], add_batch_axis=True, add_feat_axis=add_feat_axis)
inshape = sample_img.shape[1:-1]

# -------------------------
# Build model & load weights
# -------------------------
# Encoder/Decoder feature list should match training
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

model = VxmDenseSemiSupervisedSeg(
    inshape=inshape,
    nb_labels=args.nb_labels,
    nb_unet_features=[enc_nf, dec_nf],
    bidir=False,
    seg_resolution=1
)

# Load trained weights
model.load_weights(args.model)

# Get registration model
registration_model = model.get_registration_model()

# Build nearest-neighbor transform model
transform_model = vxm.networks.Transform(inshape, interp_method='nearest')

# -------------------------
# Run registration & evaluate
# -------------------------
reg_times = []
dice_means = []

for i, (img_pair, seg_pair) in enumerate(zip(img_pairs, seg_pairs)):

    moving_vol = vxm.py.utils.load_volfile(img_pair[0], add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed_vol  = vxm.py.utils.load_volfile(img_pair[1], add_batch_axis=True, add_feat_axis=add_feat_axis)
    moving_seg = vxm.py.utils.load_volfile(seg_pair[0], add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed_seg  = vxm.py.utils.load_volfile(seg_pair[1])

    start = time.time()
    warp = registration_model.predict([moving_vol, fixed_vol])
    reg_time = time.time() - start
    if i != 0:
        reg_times.append(reg_time)

    warped_seg = transform_model.predict([moving_seg, warp]).squeeze()
    overlap = vxm.py.utils.dice(warped_seg, fixed_seg, labels=np.arange(1, args.nb_labels+1))
    dice_means.append(np.mean(overlap))

    print(f'Pair {i+1}    Reg Time: {reg_time:.4f}s    Dice: {np.mean(overlap):.4f} +/- {np.std(overlap):.4f}')

print()
print(f'Avg Reg Time: {np.mean(reg_times):.4f} +/- {np.std(reg_times):.4f} (skipping first prediction)')
print(f'Avg Dice: {np.mean(dice_means):.4f} +/- {np.std(dice_means):.4f}')
