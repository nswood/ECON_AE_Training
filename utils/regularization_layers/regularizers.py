import tensorflow as tf

##############################################################################
# Custom Regularizers
##############################################################################

class DeconvOrthRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, stride=2, padding='SAME', coeff=1.0):
        """
        Args:
          stride: the stride used in the convolution (an integer).
          padding: 'SAME' or 'VALID' (note: in PyTorch code padding was an integer).
          coeff: a multiplicative coefficient for the regularization loss.
        """
        self.stride = stride
        self.padding = padding
        self.coeff = coeff

    def __call__(self, kernel):
        """
        Expects kernel to be a 4D tensor with shape:
            (kernel_height, kernel_width, in_channels, out_channels)
        and returns the norm of (conv(kernel, kernel) - target) where target
        is a tensor with ones on the “center” diagonal.
        """
        # First, convert the Keras kernel to the “PyTorch” ordering:
        # PyTorch expects (out_channels, in_channels, kernel_height, kernel_width)
        kernel_pt = tf.transpose(kernel, perm=[3, 2, 0, 1])

        # In the original PyTorch code the kernel is used both as the “input”
        # and the convolution “filter”. In TF, conv2d expects inputs of shape
        # [batch, height, width, channels]. So we first rearrange kernel_pt to
        # form an input tensor.
        inp = tf.transpose(kernel_pt, perm=[0, 2, 3, 1])  # shape: (out_channels, kH, kW, in_channels)

        # The filter for tf.nn.conv2d must have shape:
        # (filter_height, filter_width, in_channels, out_channels). We can get
        # that from kernel_pt by transposing appropriately.
        filt = tf.transpose(kernel_pt, perm=[2, 3, 1, 0])  # shape: (kH, kW, in_channels, out_channels)

        # Perform the convolution.
        conv_out = tf.nn.conv2d(inp, filt,
                                strides=[1, self.stride, self.stride, 1],
                                padding=self.padding)
        # conv_out now has shape [out_channels, new_height, new_width, out_channels].
        # Rearrange it to match the PyTorch version: (out_channels, out_channels, new_height, new_width)
        conv_out = tf.transpose(conv_out, perm=[0, 3, 1, 2])

        # Build the target tensor: zeros everywhere except that at the spatial
        # center location (ct,ct) we want the identity matrix.
        out_channels = kernel.shape[-1]  # statically known number of output channels
        shp = tf.shape(conv_out)
        new_height, new_width = shp[2], shp[3]
        target = tf.zeros((out_channels, out_channels, new_height, new_width), dtype=kernel.dtype)
        # Determine the center (using new_width; you could also use new_height)
        ct = new_width // 2

        # For each channel index i, we want target[i, i, ct, ct] = 1.
        indices = tf.stack([
            tf.range(out_channels),
            tf.range(out_channels),
            tf.fill([out_channels], ct),
            tf.fill([out_channels], ct)
        ], axis=1)  # shape: (out_channels, 4)
        updates = tf.ones([out_channels], dtype=kernel.dtype)
        target = tf.tensor_scatter_nd_update(target, indices, updates)

        # The regularization loss is the norm of (conv_out - target)
        reg_loss = tf.norm(conv_out - target)
        return self.coeff * reg_loss

    def get_config(self):
        return {'stride': self.stride, 'padding': self.padding, 'coeff': self.coeff}


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, coeff=1.0):
        """
        Args:
          coeff: a multiplicative coefficient for the regularization loss.
        """
        self.coeff = coeff

    def __call__(self, weight_matrix):
        """
        Reshapes weight_matrix to 2D and returns
            || W^T W - I ||
        """
        # Get the (possibly multi-dimensional) weight shape and reshape
        shp = tf.shape(weight_matrix)
        mat = tf.reshape(weight_matrix, (shp[0], -1))  # shape: (dim1, dim2)
        # If there are more columns than rows, transpose so that we have “tall” matrix.
        # (If static shapes are available, you can check them; here we use tf.cond for generality.)
        mat = tf.cond(tf.less(tf.shape(mat)[0], tf.shape(mat)[1]),
                      lambda: tf.transpose(mat),
                      lambda: mat)
        # Compute (W^T W) and subtract the identity.
        prod = tf.matmul(tf.transpose(mat), mat)
        dim = tf.shape(prod)[0]
        ident = tf.eye(dim, dtype=weight_matrix.dtype)
        reg_loss = tf.norm(prod - ident)
        return self.coeff * reg_loss

    def get_config(self):
        return {'coeff': self.coeff}
