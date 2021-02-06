import tensorflow as tf
import numpy as np
from cv2 import *

def conv_2d_layer(self,new_input,filter_shape,activate=tf.identity,stride=1,padding='SAME',name=None):
        with tf.variable_scope(name):
            W = tf.get_variable("W",shape=filter_shape,initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable("b",shape=filter_shape[-1],initializer=tf.constant_initializer(0.))
            conv = tf.nn.conv2d(input=new_input,filters=w,strides=[1,stride,stride,1], padding=padding)
            bias = activate(tf.nn.bias_add(value=conv,bias=b))
        return bias

def deconv_2d_layer(self,new_input,filter_shape,output_shape,activate=tf.identity,stride=1,padding='SAME',name=None):
        with tf.variable_scope(name):
            W = tf.get_variable("W",shape=filter_shape,initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable("b",shape=filter_shape[-2],initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose(input=new_input,filters=W,output_shape=output_shape,strides=[1,stride,stride,1], padding=padding)
            bias = activate(tf.nn.bias_add(value=deconv,bias=b))
        return bias
def new_fc_layer(self,new_input,output_size,name):
        shape = new_input.get_shape().as_list()
        dim = np.prod(shape[1:])
        x = tf.reshape(bottom,[-1, dim])
        input_size = dim
        with tf.variable_scope(name):
            W = tf.get_variable("W",shape=[input_size, output_size],initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable("b",shape=[output_size],initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add(value=tf.matmul(x,W),bias=b)
        return fc
        
def channel_fc_layer(self, input, name):
        _,width,height,n_feat_map = input.get_shape().as_list()
        input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )
        with tf.variable_scope(name):
            W = tf.get_variable("W",shape=[n_feat_map,width*height, width*height],initializer=tf.random_normal_initializer(0., 0.005))
            output = tf.batch_matmul(input_transpose, W)
        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape(output_transpose,[-1, height, width, n_feat_map])
        return output_reshape