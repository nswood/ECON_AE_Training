#!/usr/bin/env python
"""
compare_deconv_orth.py

This script implements the deconvolution-orthogonality and orthogonality
distance functions in both PyTorch and TensorFlow and compares their outputs
on random data.

The orthogonal convolution regularizer is implemented in torch by the authors Wang, J. et al in Orthogonal Convolutional Neural Networks: https://arxiv.org/pdf/1911.12207
"""

"""
Test outputs:
deconv_orth_dist:
  PyTorch result:     24.657180786132812
  TensorFlow result:  24.65718

=== Testing orth_dist ===
orth_dist:
  PyTorch result:     77.49185180664062
  TensorFlow result:  77.49186

"""

import numpy as np

# ----- PyTorch implementation -----
import torch

def deconv_orth_dist_torch(kernel, stride=2, padding=1):
    """
    Computes the deconvolution-orthogonality distance.
    
    Args:
      kernel: a torch.Tensor of shape (out_channels, in_channels, kernel_h, kernel_w).
      stride: stride used in conv2d.
      padding: padding used in conv2d.
    
    Returns:
      A scalar torch.Tensor representing the norm ||conv(kernel, kernel) - target||.
    """
    # Unpack dimensions (note: here kernel is used as both input and filter)
    o_c, i_c, w, h = kernel.shape
    # In torch, conv2d expects input shape (N, C, H, W) and weight shape (out_channels, in_channels, kh, kw)
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    # Build target: zeros everywhere except an identity at the center spatial location.
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1]),
                         dtype=kernel.dtype, device=kernel.device)
    ct = int(np.floor(output.shape[-1] / 2))
    target[:, :, ct, ct] = torch.eye(o_c, dtype=kernel.dtype, device=kernel.device)
    return torch.norm(output - target)

def orth_dist_torch(mat):
    """
    Computes ||W^T W - I|| for a weight matrix.
    
    Args:
      mat: a torch.Tensor that will be reshaped to 2D.
      
    Returns:
      A scalar torch.Tensor representing the orthogonality distance.
    """
    mat = mat.reshape((mat.shape[0], -1))
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1, 0)
    I = torch.eye(mat.shape[1], dtype=mat.dtype, device=mat.device)
    return torch.norm(torch.t(mat) @ mat - I)


# ----- TensorFlow implementation -----
import tensorflow as tf

def deconv_orth_dist_tf(kernel, stride=2, padding='SAME'):
    """
    Computes the deconvolution-orthogonality distance in TensorFlow.
    
    Args:
      kernel: a tf.Tensor of shape (kernel_h, kernel_w, in_channels, out_channels)
              (i.e. Keras ordering).
      stride: stride for tf.nn.conv2d (an integer).
      padding: padding for tf.nn.conv2d ('SAME' or 'VALID').
    
    Returns:
      A scalar tf.Tensor representing the norm ||conv(kernel, kernel) - target||.
    """
    # Convert to “PyTorch ordering” (out_channels, in_channels, kh, kw)
    kernel_pt = tf.transpose(kernel, perm=[3, 2, 0, 1])
    # Build an input tensor for tf.nn.conv2d. tf.nn.conv2d expects [batch, H, W, channels]
    inp = tf.transpose(kernel_pt, perm=[0, 2, 3, 1])  # shape: (out_channels, kh, kw, in_channels)
    # The filter must have shape [filter_height, filter_width, in_channels, out_channels]
    filt = tf.transpose(kernel_pt, perm=[2, 3, 1, 0])   # shape: (kh, kw, in_channels, out_channels)
    conv_out = tf.nn.conv2d(inp, filt, strides=[1, stride, stride, 1], padding=padding)
    # Rearrange output to (out_channels, out_channels, new_h, new_w)
    conv_out = tf.transpose(conv_out, perm=[0, 3, 1, 2])
    
    # Create the target tensor: zeros everywhere except for an identity at the spatial center.
    out_channels = tf.shape(kernel)[-1]
    new_h = tf.shape(conv_out)[2]
    new_w = tf.shape(conv_out)[3]
    target = tf.zeros((out_channels, out_channels, new_h, new_w), dtype=kernel.dtype)
    ct = new_w // 2  # center column (assumes square spatial dimensions)
    indices = tf.stack([
        tf.range(out_channels),
        tf.range(out_channels),
        tf.fill([out_channels], ct),
        tf.fill([out_channels], ct)
    ], axis=1)
    updates = tf.ones([out_channels], dtype=kernel.dtype)
    target = tf.tensor_scatter_nd_update(target, indices, updates)
    
    diff = conv_out - target
    return tf.norm(diff)

def orth_dist_tf(mat):
    """
    Computes ||W^T W - I|| for a weight matrix in TensorFlow.
    
    Args:
      mat: a tf.Tensor that will be reshaped to 2D.
      
    Returns:
      A scalar tf.Tensor representing the orthogonality distance.
    """
    shape = tf.shape(mat)
    mat_flat = tf.reshape(mat, (shape[0], -1))
    # If the number of rows is less than the number of columns, transpose.
    cond = tf.less(tf.shape(mat_flat)[0], tf.shape(mat_flat)[1])
    mat_flat = tf.cond(cond, lambda: tf.transpose(mat_flat), lambda: mat_flat)
    prod = tf.matmul(tf.transpose(mat_flat), mat_flat)
    dim = tf.shape(prod)[0]
    I = tf.eye(dim, dtype=mat.dtype)
    return tf.norm(prod - I)


# ----- Main testing code -----
def main():
    # Set seeds for reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)

    print("=== Testing deconv_orth_dist ===")
    # Create random kernel data in numpy.
    # Torch expects kernel of shape (out_channels, in_channels, kernel_h, kernel_w).
    out_channels = 4
    in_channels = 3
    kernel_size = 3
    kernel_np = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)
    
    # --- PyTorch ---
    kernel_torch = torch.tensor(kernel_np)
    result_torch = deconv_orth_dist_torch(kernel_torch).item()
    
    # --- TensorFlow ---
    # TF (and Keras) expects kernels in shape (kernel_h, kernel_w, in_channels, out_channels).
    kernel_tf = tf.convert_to_tensor(np.transpose(kernel_np, (2, 3, 1, 0)))
    result_tf = deconv_orth_dist_tf(kernel_tf).numpy()
    
    print("deconv_orth_dist:")
    print("  PyTorch result:    ", result_torch)
    print("  TensorFlow result: ", result_tf)
    
    print("\n=== Testing orth_dist ===")
    # Create a random matrix.
    mat_shape = (10, 20)
    mat_np = np.random.randn(*mat_shape).astype(np.float32)
    
    # --- PyTorch ---
    mat_torch = torch.tensor(mat_np)
    result_torch_orth = orth_dist_torch(mat_torch).item()
    
    # --- TensorFlow ---
    mat_tf = tf.convert_to_tensor(mat_np)
    result_tf_orth = orth_dist_tf(mat_tf).numpy()
    
    print("orth_dist:")
    print("  PyTorch result:    ", result_torch_orth)
    print("  TensorFlow result: ", result_tf_orth)

if __name__ == "__main__":
    main()
