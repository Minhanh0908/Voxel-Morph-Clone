#!/usr/bin/env python
"""
Train VoxelMorph (atlas-based) using unsupervised loss:
    L_us = L_sim (MSE or NCC) + λ * L_smooth
"""

import os, argparse, numpy as np, tensorflow as tf, voxelmorph as vxm

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

# -------------------------
# Parse arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img-list', required=True, help='list of training images (moving scans)')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', required=True, help='path to atlas volume')
parser.add_argument('--model-dir', default='models_us_atlas', help='directory to save models')
parser.add_argument('--gpu', default='0', help='GPU ID')
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--steps-per-epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lambda', dest='lambda_weight', type=float, default=0.01)
parser.add_argument('--image-loss', default='mse', help='mse or ncc')
args = parser.parse_args()

# -------------------------
# GPU setup
# -------------------------
device, nb_devices = vxm.utils.setup_device(args.gpu)

# -------------------------
# Load data
# -------------------------
train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)

atlas = vxm.py.utils.load_volfile(args.atlas, add_batch_axis=True, add_feat_axis=True)

# Generator: scan-to-atlas
generator = vxm.generators.scan_to_atlas(
    train_files, atlas, batch_size=1, bidir=False, add_feat_axis=True
)

# Get shape info
sample_shape = next(generator)[0][0].shape
inshape = sample_shape[1:-1]
nfeats = sample_shape[-1]

# -------------------------
# Build model
# -------------------------
model = vxm.networks.VxmDense(
    inshape=inshape,
    src_feats=nfeats,
    trg_feats=nfeats,
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
losses = [sim_loss, grad_loss]
weights = [1.0, args.lambda_weight]

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
