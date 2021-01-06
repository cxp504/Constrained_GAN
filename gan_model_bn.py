# coding=utf-8
# DCGAN network, BN in G and D
import tensorflow as tf

batchsize = 64
def lrelu(input_, leak=0.2, name="lrelu"):
    return tf.maximum(input_, leak * input_, name=name)

def deconv(inputs, shape, strides, out_num):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    #bias = tf.get_variable("bias", shape=[shape[-2]], initializer=tf.constant_initializer([0]))
    return tf.nn.conv2d_transpose(inputs, filters, out_num, strides)

#SAME means padding=1
def conv(inputs, shape, strides, use_sn=False):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    #bias = tf.get_variable("bias", shape=[shape[-1]], initializer=tf.constant_initializer([0]))
    if use_sn:
        return tf.nn.conv2d(inputs, spectral_norm("sn", filters), strides, "SAME") 
    else:
        return tf.nn.conv2d(inputs, filters, strides, "SAME") 

def fully_connected(inputs, num_out, use_sn=False):
    W = tf.get_variable("W", [inputs.shape[-1], num_out], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0]))
    if use_sn:
        return tf.matmul(inputs, spectral_norm("sn", W)) + b
    else:
        return tf.matmul(inputs, W) + b

def spectral_norm(name, w, iteration=1):
    print("spectral_norm")
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm

def batch_norm(input_, is_training, scope, epsilon=1e-5):
    print("batch_norm")
    return tf.contrib.layers.batch_norm(
        input_,
        decay=0.9,
        updates_collections=None,
        epsilon=epsilon,
        scale=True,
        fused=False,
        is_training=is_training,
        scope=scope)

def layer_norm(input_, is_training, scope):
    print("layer_norm")
    return tf.contrib.layers.layer_norm(
        input_, trainable=is_training, scope=scope)

def Generator(Z, is_training=True, reuse=False):
    with tf.variable_scope(name_or_scope="generator", reuse=reuse):
        with tf.variable_scope(name_or_scope="linear"):
            inputs = tf.reshape(fully_connected(Z, 4 * 4 * 128), [batchsize, 4, 4, 128])
        with tf.variable_scope(name_or_scope="deconv1"):
            inputs = deconv(inputs, [4, 4, 1024, 128], [1, 1, 1, 1], [batchsize, 4, 4, 1024])
            inputs = batch_norm(inputs, is_training=is_training, scope='g_bn1', epsilon=1e-5)
            inputs = tf.nn.relu(inputs)
        with tf.variable_scope(name_or_scope="deconv2"):
            inputs = deconv(inputs, [4, 4, 512, 1024], [1, 2, 2, 1], [batchsize, 8, 8, 512])
            inputs = batch_norm(inputs, is_training=is_training, scope='g_bn2', epsilon=1e-5)
            inputs = tf.nn.relu(inputs)
        with tf.variable_scope(name_or_scope="deconv3"):
            inputs = deconv(inputs, [4, 4, 256, 512], [1, 2, 2, 1], [batchsize, 16, 16, 256])
            inputs = batch_norm(inputs, is_training=is_training, scope='g_bn3', epsilon=1e-5)
            inputs = tf.nn.relu(inputs)
        with tf.variable_scope(name_or_scope="deconv4"):
            inputs = deconv(inputs, [4, 4, 128, 256], [1, 2, 2, 1], [batchsize, 32, 32, 128])
            inputs = batch_norm(inputs, is_training=is_training, scope='g_bn4', epsilon=1e-5)
            inputs = tf.nn.relu(inputs)
        with tf.variable_scope(name_or_scope="deconv5"):
            inputs = deconv(inputs, [4, 4, 64, 128], [1, 2, 2, 1], [batchsize, 64, 64, 64])
            inputs = batch_norm(inputs, is_training=is_training, scope='g_bn5', epsilon=1e-5)
            inputs = tf.nn.relu(inputs)
        with tf.variable_scope(name_or_scope="deconv6"):
            inputs = deconv(inputs, [4, 4, 3, 64], [1, 2, 2, 1], [batchsize, 128, 128, 3])
            inputs = tf.nn.tanh(inputs)
        return inputs

def Discriminator(inputs, is_training=None, reuse=False):
    use_sn = False
    with tf.variable_scope(name_or_scope="discriminator", reuse=reuse):
        with tf.variable_scope("conv1"):
            inputs = conv(inputs, [4, 4, 3, 64], [1, 2, 2, 1], use_sn=use_sn)
            inputs = lrelu(inputs)
        with tf.variable_scope("conv2"):
            inputs = conv(inputs, [4, 4, 64, 128], [1, 2, 2, 1], use_sn=use_sn)
            inputs = batch_norm(inputs, is_training=is_training, scope='d_bn2', epsilon=1e-5)
            inputs = lrelu(inputs)
        with tf.variable_scope("conv3"):
            inputs = conv(inputs, [4, 4, 128, 256], [1, 2, 2, 1], use_sn=use_sn)
            inputs = batch_norm(inputs, is_training=is_training, scope='d_bn3', epsilon=1e-5)
            inputs = lrelu(inputs)
        with tf.variable_scope("conv4"):
            inputs = conv(inputs, [4, 4, 256, 512], [1, 2, 2, 1], use_sn=use_sn)
            inputs = batch_norm(inputs, is_training=is_training, scope='d_bn4', epsilon=1e-5)
            inputs = lrelu(inputs)
        with tf.variable_scope("conv5"):
            inputs = conv(inputs, [4, 4, 512, 1024], [1, 2, 2, 1], use_sn=use_sn)
            inputs = batch_norm(inputs, is_training=is_training, scope='d_bn5', epsilon=1e-5)
            inputs = lrelu(inputs)
        with tf.variable_scope("conv6"):
            inputs = conv(inputs, [4, 4, 1024, 1], [1, 2, 2, 1], use_sn=use_sn)
        return tf.nn.sigmoid(inputs), inputs
