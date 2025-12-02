#!/usr/bin/env python
"""
Train VoxelMorph (scan-to-scan) with segmentation supervision using:
    L_total = L_image + λ * L_grad + α * L_seg
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

# disable eager execution for VoxelMorph v1
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# -------------------------
# Parse arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img-list', required=True, help='list of moving images')
parser.add_argument('--seg-list', required=True, help='list of segmentation labels')
parser.add_argument('--nb-labels', type=int, required=True, help='number of segmentation labels')
parser.add_argument('--model-dir', default='models_seg_scan')
parser.add_argument('--gpu', default='0')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--steps-per-epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lambda', dest='lambda_weight', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--image-loss', default='mse', help='mse or ncc')
args = parser.parse_args()

# -------------------------
# GPU setup
# -------------------------
device, nb_devices = vxm.utils.setup_device(args.gpu)
print(f"Using device: {device}, #GPUs: {nb_devices}")

# -------------------------
# Load data
# -------------------------
train_files = vxm.py.utils.read_file_list(args.img_list)
seg_files   = vxm.py.utils.read_file_list(args.seg_list)
assert len(train_files) == len(seg_files), "Image and segmentation list must match"

# -------------------------
# Generator with segmentation supervision
# -------------------------
generator = vxm.generators.semisupervised(
    vol_names=train_files,
    seg_names=seg_files,
    labels=np.arange(args.nb_labels),
    downsize=1  # keep segmentation full size
)

# sample shape
sample = next(generator)
inshape = sample[0][0].shape[1:-1]  # remove batch axis

print("Input shape:", inshape)
print("Training model with scan-to-scan + segmentation supervision...")

# -------------------------
# Build model
# -------------------------
model = vxm.networks.VxmDenseSemiSupervisedSeg(
    inshape=inshape,
    nb_labels=args.nb_labels,
    seg_resolution=1,  # full size seg
    bidir=False
)

# -------------------------
# Define losses
# -------------------------
if args.image_loss.lower() == 'ncc':
    sim_loss = vxm.losses.NCC().loss
else:
    sim_loss = vxm.losses.MSE().loss

grad_loss = vxm.losses.Grad('l2', loss_mult=1).loss
dice_loss = vxm.losses.Dice().loss

losses = [sim_loss, grad_loss, dice_loss]
weights = [1.0, args.lambda_weight, args.alpha]

# -------------------------
# Compile model
# -------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
    loss=losses,
    loss_weights=weights
)

# -------------------------
# Setup checkpoints
# -------------------------
os.makedirs(args.model_dir, exist_ok=True)
save_path = os.path.join(args.model_dir, '{epoch:04d}.h5')
save_callback = tf.keras.callbacks.ModelCheckpoint(save_path, save_freq=20*args.steps_per_epoch)

# -------------------------
# Train
# -------------------------
model.fit(
    generator,
    epochs=args.epochs,
    steps_per_epoch=args.steps_per_epoch,
    callbacks=[save_callback],
    verbose=1
)
