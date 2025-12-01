#!/usr/bin/env python
"""
Train VoxelMorph (atlas-based) with segmentation supervision:
    L_alpha = L_us + α * L_seg
"""

import os, argparse, numpy as np, tensorflow as tf, voxelmorph as vxm

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# -------------------------
# Parse arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img-list', required=True, help='list of training images (moving scans)')
parser.add_argument('--seg-list', required=True, help='list of segmentation labels for moving scans')
parser.add_argument('--atlas', required=True, help='path to atlas image')
parser.add_argument('--atlas-seg', required=True, help='path to atlas segmentation')
parser.add_argument('--nb-labels', type=int, required=True, help='number of segmentation labels')
parser.add_argument('--model-dir', default='models_seg_atlas')
parser.add_argument('--gpu', default='0')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--steps-per-epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lambda', dest='lambda_weight', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--image-loss', default='mse', help='mse or ncc')
args = parser.parse_args()

# -------------------------
# GPU setup
# -------------------------
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# -------------------------
# Load data
# -------------------------
train_files = vxm.py.utils.read_file_list(args.img_list)
seg_files = vxm.py.utils.read_file_list(args.seg_list)
atlas = vxm.py.utils.load_volfile(args.atlas, add_batch_axis=True, add_feat_axis=True)
atlas_seg = vxm.py.utils.load_volfile(args.atlas_seg, add_batch_axis=True, add_feat_axis=True)
assert len(train_files) == len(seg_files), "Image and segmentation list lengths must match."

# Generator with segmentation (custom version)
generator = vxm.generators.scan_to_atlas_with_seg(
    train_files, seg_files, atlas, atlas_seg, batch_size=1
)

# Shape info
sample = next(generator)
inshape = sample[0][0].shape[1:-1]

# -------------------------
# Build model
# -------------------------
model = vxm.networks.VxmDenseSemiSupervisedSeg(
    inshape=inshape,
    nb_labels=args.nb_labels,
    bidir=False
)

# -------------------------
# Define losses
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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
              loss=losses, loss_weights=weights)

os.makedirs(args.model_dir, exist_ok=True)
save_path = os.path.join(args.model_dir, '{epoch:04d}.h5')
save_callback = tf.keras.callbacks.ModelCheckpoint(save_path, save_freq=20*args.steps_per_epoch)

model.fit(generator,
          epochs=args.epochs,
          steps_per_epoch=args.steps_per_epoch,
          callbacks=[save_callback],
          verbose=1)
