#!/usr/bin/env python
"""
Train VoxelMorph (scan-to-scan) with segmentation supervision:
    L_alpha = L_us + α * L_seg
    where L_us = L_sim + λ * L_smooth
"""

import os
import argparse
import warnings
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

# Disable eager execution for TF1 compatibility
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# -------------------------
# Parse arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img-list', required=True, help='list of training images (NIfTI paths)')
parser.add_argument('--seg-list', required=True, help='list of segmentation labels (matching order)')
parser.add_argument('--model-dir', default='models_seg_scan', help='directory to save model weights')
parser.add_argument('--gpu', default='0', help='GPU ID (default: 0)')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--steps-per-epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lambda', dest='lambda_weight', type=float, default=0.01, help='smoothness weight')
parser.add_argument('--alpha', type=float, default=1.0, help='segmentation loss weight')
parser.add_argument('--nb-labels', type=int, required=True, help='number of segmentation labels')
parser.add_argument('--image-loss', default='mse', help='mse or ncc')
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decoder filters')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps')
parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration')
args = parser.parse_args()

# -------------------------
# Device setup
# -------------------------
device, nb_devices = vxm.utils.setup_device(args.gpu)

# -------------------------
# Load data
# -------------------------
img_files = vxm.py.utils.read_file_list(args.img_list)
seg_files = vxm.py.utils.read_file_list(args.seg_list)
assert len(img_files) == len(seg_files), "Number of images and segmentations must match."

# -------------------------
# Generator: scan-to-scan with segmentation
# -------------------------
def scan_to_scan_with_seg(img_files, seg_files, batch_size=1, nb_labels=4):
    """
    Generator for scan-to-scan registration with segmentation supervision.
    Returns:
        inputs:  [moving_img, fixed_img, moving_seg_onehot]
        outputs: [fixed_img, fixed_seg_onehot, zeros_flow]
    """
    zeros = None
    volgen_img = vxm.generators.volgen(img_files, batch_size=batch_size)
    volgen_seg = vxm.generators.volgen(seg_files, batch_size=batch_size)

    while True:
        moving = next(volgen_img)[0].astype(np.float32)
        fixed  = next(volgen_img)[0].astype(np.float32)
        moving_seg = next(volgen_seg)[0]
        fixed_seg  = next(volgen_seg)[0]

        # One-hot encoding segmentation
        moving_seg_oh = tf.one_hot(tf.cast(moving_seg[..., 0], tf.int32), depth=nb_labels)
        fixed_seg_oh  = tf.one_hot(tf.cast(fixed_seg[..., 0], tf.int32), depth=nb_labels)

        # Convert to NumPy arrays (TF1 graph mode)
        moving_seg_oh = tf.compat.v1.keras.backend.eval(moving_seg_oh)
        fixed_seg_oh  = tf.compat.v1.keras.backend.eval(fixed_seg_oh)

        # Initialize zeros flow if needed
        if zeros is None:
            shape = moving.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)), dtype=np.float32)

        inputs = [moving, fixed, moving_seg_oh]
        outputs = [fixed, fixed_seg_oh, zeros]

        yield (inputs, outputs)

# Build generator
generator = scan_to_scan_with_seg(img_files, seg_files, batch_size=args.batch_size, nb_labels=args.nb_labels)

# Sample batch to get input shape
sample = next(generator)
inshape = sample[0][0].shape[1:-1]

# Prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# Unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# Prepare model checkpoint save path
save_filename = os.path.join(model_dir, '{epoch:04d}.h5')

# -------------------------
# Build model
# -------------------------
model = vxm.networks.VxmDenseSemiSupervisedSeg(
    inshape=inshape,
    nb_labels=args.nb_labels,
    nb_unet_features=[enc_nf, dec_nf],
    bidir=False,
    seg_resolution=1,
    int_steps=args.int_steps,
    int_downsize=args.int_downsize
)

# -------------------------
# Define loss functions
# -------------------------
if args.image_loss == 'ncc':
    sim_loss = vxm.losses.NCC().loss
else:
    sim_loss = vxm.losses.MSE().loss

grad_loss = vxm.losses.Grad('l2', loss_mult=2).loss
dice_loss = vxm.losses.Dice().loss

losses = [sim_loss, grad_loss, dice_loss]
weights = [1.0, args.lambda_weight, args.alpha]

# -------------------------
# Compile & train
# -------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
    loss=losses,
    loss_weights=weights
)

# Callback for saving model
save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename,
                                                   save_freq=20 * args.steps_per_epoch)

# Save initial weights
model.save(save_filename.format(epoch=0))

# Train model
model.fit(
    generator,
    epochs=args.epochs,
    steps_per_epoch=args.steps_per_epoch,
    callbacks=[save_callback],
    verbose=1
)
