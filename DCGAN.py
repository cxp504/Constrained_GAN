# coding=utf-8
# DCGAN network from https://github.com/google/compare_gan

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import csv
import gan_model as gan_cifar

img_H = 128
img_W = 128
img_C = 3
batchsize = 64
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    img = 2*(tf.cast(img, tf.float32) * (1. / 255) - 0.5)
    label = tf.cast(features['label'], tf.int64)
    return img, label

def mapping(x):
    return ((x + 1.0)*(255.0/2.0))

def train():
    Z = tf.placeholder(tf.float32, [batchsize, 128], name="z_float")
    real_img = tf.placeholder(tf.float32, [batchsize, img_H, img_W, img_C], name="img_float")
    fake_img = gan_cifar.Generator(Z, is_training=True, reuse=False)
    G = gan_cifar.Generator(Z, is_training=False, reuse=True)
    print('generative ready over')
    fake_out, _ = gan_cifar.Discriminator(fake_img, is_training=True, reuse=False)
    real_out, _ = gan_cifar.Discriminator(real_img, is_training=True, reuse=True)
    print('discriminator ready over')

    out_x = tf.reduce_mean(real_out)
    out_y = tf.reduce_mean(fake_out)
    h = tf.reduce_mean(tf.pow(tf.log(real_out) - tf.log(fake_out), 2)) * 0.3

    d_loss = - (tf.reduce_mean(tf.log(real_out) + tf.log(1.0 - fake_out))) + h
    g_loss = - tf.reduce_mean(tf.log(fake_out))
    t_vars = tf.trainable_variables()
    print(len(t_vars))
    print(t_vars)
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    print(len(d_vars))
    print(d_vars)
    g_vars = [var for var in t_vars if "generator" in var.name]
    print(len(g_vars))
    print(g_vars)
    opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5, beta2=0.999).minimize(d_loss, var_list=d_vars)
    opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5, beta2=0.999).minimize(g_loss, var_list=g_vars)
    images, labels = read_and_decode('./train.tfrecords')
    img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                batch_size=batchsize,
                                                num_threads=4,
                                                capacity=64000,
                                                min_after_dequeue=32000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    saver = tf.train.Saver(max_to_keep=0)
    coord = tf.train.Coordinator()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        data_path = './save_path_of_model_and_data'
        for i in range(50001):
            batch, _ = sess.run([img_batch, label_batch])
            z = np.random.standard_normal([batchsize, 128])
            d_lo, g_lo, h_lo, DX, DY = sess.run([d_loss, g_loss, h, out_x, out_y], feed_dict={real_img: batch, Z: z})
            sess.run(opt_D, feed_dict={real_img: batch, Z: z})
            sess.run(opt_G, feed_dict={real_img: batch, Z: z})
            l = []
            l.append(i)
            l.append(d_lo)
            l.append(g_lo)
            l.append(h_lo)
            l.append(DX)
            l.append(DY)
            out = open(data_path + '/out.csv', 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            csv_write.writerow(l)
            print("step: %d, d_loss: %g, g_loss: %g, h: %g, d_x: %g, d_y: %g" % (i, d_lo, g_lo, h_lo, DX, DY))
            if np.isnan(d_lo):
                    break
            if i % 1000 == 0:
                saver.save(sess, data_path + '/model/' + "model.ckpt", global_step=i)
                directory = data_path + '/data/' + str(i)
                os.makedirs(directory, exist_ok=True)
                for k in range(79):
                    z1 = np.random.standard_normal([batchsize, 128])
                    imgs = sess.run(G, feed_dict={Z: z1})
                    for j in range(batchsize):
                        if k * 64 + j >= 4000:
                                break
                        Image.fromarray(np.uint8(mapping(imgs[j, :, :, :]))).save(
                                directory + "/" + str(k * 64 + j) + ".png")
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    train()

