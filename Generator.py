# TEAM ENIGMA
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_v2_behavior()

# Class Build for the Testing Model
class Generate():
    def __init__(self):
        pass

    # Function for creating a convolutional layer along with weights:
    def conv_2d(self,input_image,filter_shape,activation=tf.identity,padding='SAME',stride=1,name=None):
        with tf.compat.v1.variable_scope(name):
            W = tf.compat.v1.get_variable("W", shape=filter_shape, initializer=tf.compat.v1.random_normal_initializer(0., 0.005))
            b = tf.compat.v1.get_variable("b", shape=filter_shape[-1], initializer=tf.compat.v1.constant_initializer(0.))
            conv = tf.nn.conv2d( input=input_image, filters=W, strides=[1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))
        return bias 

    # Function for creating a de-convolutional layer along with weights:
    def conv_2d_transpose(self,input_image, filter_shape,output_shape, activation=tf.identity,padding='SAME', stride=1, name=None):
        with tf.compat.v1.variable_scope(name):
            W = tf.compat.v1.get_variable("W", shape=filter_shape, initializer=tf.compat.v1.random_normal_initializer(0., 0.005))
            b = tf.compat.v1.get_variable("b", shape=filter_shape[-2], initializer=tf.compat.v1.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose(input_image, W, output_shape, strides=[1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))
        return bias

    def generator( self, images, is_train ):  
        with tf.compat.v1.variable_scope('GEN'):
            # Encoder part(convolution, fraction-strided convolution, ELU):
            # We have used as it has less error rates as compared to leakyReLU for larger dataset
            conv1_1 = self.conv_2d(images, [3,3,3,32], stride=1, name="conv1_1" )
            conv1_1 = tf.nn.elu(conv1_1)
            conv1_2 = self.conv_2d(conv1_1, [3,3,32,32], stride=1, name="conv1_2" )
            conv1_2 = tf.nn.elu(conv1_2)
            conv1_stride = self.conv_2d(conv1_2, [3,3,32,32], stride=2, name="conv1_stride")    # replaced max pooling by strided convolution for performance enhancement.
            conv2_1 = self.conv_2d(conv1_stride, [3,3,32,64], stride=1, name="conv2_1" )
            conv2_1 = tf.nn.elu(conv2_1)
            conv2_2 = self.conv_2d(conv2_1, [3,3,64, 64], stride=1, name="conv2_2" )
            conv2_2 = tf.nn.elu(conv2_2)
            conv2_stride = self.conv_2d(conv2_2, [3,3,64,64], stride=2, name="conv2_stride")    # replaced max pooling by strided convolution for performance enhancement.
            conv3_1 = self.conv_2d(conv2_stride, [3,3,64,128], stride=1, name="conv3_1" )
            conv3_1 = tf.nn.elu(conv3_1)
            conv3_2 = self.conv_2d(conv3_1, [3,3,128, 128], stride=1, name="conv3_2" )
            conv3_2 = tf.nn.elu(conv3_2)
            conv3_3 = self.conv_2d(conv3_2, [3,3,128,128], stride=1, name="conv3_3" )
            conv3_3 = tf.nn.elu(conv3_3)
            conv3_4 = self.conv_2d(conv3_3, [3,3,128, 128], stride=1, name="conv3_4" )   
            conv3_4 = tf.nn.elu(conv3_4)
            conv3_stride = self.conv_2d(conv3_4, [3,3,128,128], stride=2, name="conv3_stride")   # replaced max pooling by strided convolution for performance enhancement.
            conv4_stride = self.conv_2d(conv3_stride, [3,3,128,128], stride=2, name="conv4_stride")   # replaced max pooling by strided convolution for performance enhancement.
            conv4_stride = tf.nn.elu(conv4_stride)
            conv5_stride = self.conv_2d(conv4_stride, [3,3,128,128], stride=2, name="conv5_stride")    # replaced max pooling by strided convolution for performance enhancement.
            conv5_stride = tf.nn.elu(conv5_stride)
            conv6_stride = self.conv_2d(conv5_stride, [3,3,128,128], stride=2, name="conv6_stride") 
            conv6_stride = tf.nn.elu(conv6_stride)
 
            # Decoder part
            deconv5_fs = self.conv_2d_transpose( conv6_stride, [3,3,128,128], conv5_stride.get_shape().as_list(), stride=2, name="deconv5_fs")
            debn5_fs = tf.nn.elu(deconv5_fs)
            skip5 = tf.concat([debn5_fs, conv5_stride], 3)
            channels5 = skip5.get_shape().as_list()[3]
            deconv4_fs = self.conv_2d_transpose( skip5, [3,3,128,channels5], conv4_stride.get_shape().as_list(), stride=2, name="deconv4_fs")
            debn4_fs = tf.nn.elu(deconv4_fs)
            skip4 = tf.concat([debn4_fs, conv4_stride], 3)
            channels4 = skip4.get_shape().as_list()[3]
            deconv3_fs = self.conv_2d_transpose( skip4, [3,3,128,channels4], conv3_stride.get_shape().as_list(), stride=2, name="deconv3_fs")
            debn3_fs = tf.nn.elu(deconv3_fs)
            skip3 = tf.concat([debn3_fs, conv3_stride], 3)
            channels3 = skip3.get_shape().as_list()[3]
            deconv2_fs = self.conv_2d_transpose( skip3, [3,3,64,channels3], conv2_stride.get_shape().as_list(), stride=2, name="deconv2_fs")
            debn2_fs = tf.nn.elu(deconv2_fs)
            skip2 = tf.concat([debn2_fs, conv2_stride], 3)
            channels2 = skip2.get_shape().as_list()[3]
            deconv1_fs = self.conv_2d_transpose( skip2, [3,3,32,channels2], conv1_stride.get_shape().as_list(), stride=2, name="deconv1_fs")
            debn1_fs = tf.nn.elu(deconv1_fs)  
            skip1 = tf.concat([debn1_fs, conv1_stride], 3)
            channels1 = skip1.get_shape().as_list()[3]
            recon = self.conv_2d_transpose( skip1, [3,3,3,channels1],  images.get_shape().as_list(), stride=2, name="recon") 
        return recon
  
# TEAM ENIGMA
