#!/usr/bin/env python
"""
Register two volumes using a semi-supervised VxmDenseSemiSupervisedSeg model.
Saves the warped image and optional warp field.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
from voxelmorph.networks import VxmDenseSemiSupervisedSeg

# -------------------------
# Parse arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('--model', required=True, help='trained semi-supervised model (.h5)')
parser.add_argument('--nb-labels', type=int, required=True, help='number of segmentation labels in model')
parser.add_argument('--gpu', help='GPU number(s), default CPU if not provided')
parser.add_argument('--multichannel', action='store_true', help='specify multi-channel images')
args = parser.parse_args()

# -------------------------
# Device handling
# -------------------------
device, nb_devices = vxm.utils.setup_device(args.gpu)
add_feat_axis = not args.multichannel

# -------------------------
# Load images
# -------------------------
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

inshape = moving.shape[1:-1]
nb_feats = moving.shape[-1]

# -------------------------
# Build model and load weights
# -------------------------
# Note: nb_unet_features phải trùng với khi train, bạn chỉnh nếu khác
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

with tf.device(device):
    model = VxmDenseSemiSupervisedSeg(
        inshape=inshape,
        nb_labels=args.nb_labels,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=False,
        seg_resolution=1
    )
    model.load_weights(args.model)

    # Get registration model
    registration_model = model.get_registration_model()
    warp = registration_model.predict([moving, fixed])

    # Transform moving image
    moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

# -------------------------
# Save results
# -------------------------
if args.warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)

print(f"Registration done! Warped image saved at {args.moved}")
if args.warp:
    print(f"Warp field saved at {args.warp}")
