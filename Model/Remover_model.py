import tensorflow as tf
import numpy as np
tf.compat.v1.disable_v2_behavior()

# Build a class for model
class Reconstruct():
    def __init__(self):
        pass

    def conv_2d_layer(self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable("W", shape=filter_shape, initializer=tf.compat.v1.random_normal_initializer(0., 0.005))
            b = tf.compat.v1.get_variable("b", shape=filter_shape[-1], initializer=tf.compat.v1.constant_initializer(0.))
            conv = tf.nn.conv2d( input=bottom, filters=w, strides=[1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias 

    def conv_2d_transpose(self, bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.compat.v1.variable_scope(name):
            W = tf.compat.v1.get_variable("W", shape=filter_shape, initializer=tf.compat.v1.random_normal_initializer(0., 0.005))
            b = tf.compat.v1.get_variable("b", shape=filter_shape[-2], initializer=tf.compat.v1.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias
    def resize_conv_layer(self, bottom, filter_shape, resize_scale=2, activation=tf.identity, padding='SAME', stride=1, name=None):
        width = bottom.get_shape().as_list()[1]
        height = bottom.get_shape().as_list()[2]
        bottom = tf.image.resize(bottom, [width*resize_scale, height*resize_scale], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        bias = self.conv_2d_layer(bottom, filter_shape, stride=1, name=name )
        return bias

    def new_fc_layer( self, bottom, output_size, name ):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])
        input_size = dim

        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable("W", shape=[input_size, output_size], initializer=tf.compat.v1.random_normal_initializer(0., 0.005))
            b = tf.compat.v1.get_variable("b", shape=[output_size], initializer=tf.compat.v1.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b)

        return fc

    def leaky_relu(self,new_input):
        return tf.nn.leaky_relu(new_input,alpha=0.1,)

    def batchnorm(self, bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        with tf.compat.v1.variable_scope(name):
            gamma = tf.compat.v1.get_variable("gamma", [depth], initializer=tf.compat.v1.constant_initializer(1.))
            beta  = tf.compat.v1.get_variable("beta" , [depth], initializer=tf.compat.v1.constant_initializer(0.))

            batch_mean, batch_var = tf.nn.moments(x=bottom, axes=[0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            mean, var = tf.cond(pred=is_train, true_fn=update, false_fn=lambda: (ema_mean, ema_var) )
            normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)
        
        return normed

    def channel_wise_fc_layer(self, input, name): # bottom: (7x7x512)
        _, width, height, n_feat_map = input.get_shape().as_list()
        input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( a=input_reshape, perm=[2,0,1] )

        with tf.compat.v1.variable_scope(name):
            # (512,49,49)
            W = tf.compat.v1.get_variable("W", shape=[n_feat_map,width*height, width*height], initializer=tf.compat.v1.random_normal_initializer(0., 0.005))
            output = tf.batch_matmul(input_transpose, W)

        output_transpose = tf.transpose(a=output, perm=[1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape
    
  

    def generator( self, images, is_train ):  
        with tf.compat.v1.variable_scope('GEN'):
  
            # VGG 19 for Feature learning
            # 1.
            conv1_1 = self.conv_2d_layer(images, [3,3,3,32], stride=1, name="conv1_1" )
            conv1_1 = tf.nn.elu(conv1_1)
            conv1_2 = self.conv_2d_layer(conv1_1, [3,3,32,32], stride=1, name="conv1_2" )
            conv1_2 = tf.nn.elu(conv1_2)
            # Use stride convolution to replace max pooling (with padding to keep retain size 128->64)
            conv1_stride = self.conv_2d_layer(conv1_2, [3,3,32,32], stride=2, name="conv1_stride")
            
            # 2.
            conv2_1 = self.conv_2d_layer(conv1_stride, [3,3,32,64], stride=1, name="conv2_1" )
            conv2_1 = tf.nn.elu(conv2_1)
            conv2_2 = self.conv_2d_layer(conv2_1, [3,3,64, 64], stride=1, name="conv2_2" )
            conv2_2 = tf.nn.elu(conv2_2)
            # Use stride convolution to replace max pooling (with padding to keep retain size 64->32)
            conv2_stride = self.conv_2d_layer(conv2_2, [3,3,64,64], stride=2, name="conv2_stride")
            
            # 3.
            conv3_1 = self.conv_2d_layer(conv2_stride, [3,3,64,128], stride=1, name="conv3_1" )
            conv3_1 = tf.nn.elu(conv3_1)
            conv3_2 = self.conv_2d_layer(conv3_1, [3,3,128, 128], stride=1, name="conv3_2" )
            conv3_2 = tf.nn.elu(conv3_2)
            conv3_3 = self.conv_2d_layer(conv3_2, [3,3,128,128], stride=1, name="conv3_3" )
            conv3_3 = tf.nn.elu(conv3_3)
            conv3_4 = self.conv_2d_layer(conv3_3, [3,3,128, 128], stride=1, name="conv3_4" )   
            conv3_4 = tf.nn.elu(conv3_4)
            # Use stride convolution to replace max pooling (with padding to keep retain size 32->16)
            conv3_stride = self.conv_2d_layer(conv3_4, [3,3,128,128], stride=2, name="conv3_stride") # Final feature map (temporary)
            
            conv4_stride = self.conv_2d_layer(conv3_stride, [3,3,128,128], stride=2, name="conv4_stride") # 16 -> 8
            conv4_stride = tf.nn.elu(conv4_stride)
            
            conv5_stride = self.conv_2d_layer(conv4_stride, [3,3,128,128], stride=2, name="conv5_stride") # 8 -> 4
            conv5_stride = tf.nn.elu(conv5_stride)
            
            conv6_stride = self.conv_2d_layer(conv5_stride, [3,3,128,128], stride=2, name="conv6_stride") # 4 -> 1
            conv6_stride = tf.nn.elu(conv6_stride)
 
            # 6.
            deconv5_fs = self.conv_2d_transpose( conv6_stride, [3,3,128,128], conv5_stride.get_shape().as_list(), stride=2, name="deconv5_fs")
            debn5_fs = tf.nn.elu(deconv5_fs)
            
            skip5 = tf.concat([debn5_fs, conv5_stride], 3)
            channels5 = skip5.get_shape().as_list()[3]
            
            # 5.    
            deconv4_fs = self.conv_2d_transpose( skip5, [3,3,128,channels5], conv4_stride.get_shape().as_list(), stride=2, name="deconv4_fs")
            debn4_fs = tf.nn.elu(deconv4_fs)
            
            skip4 = tf.concat([debn4_fs, conv4_stride], 3)
            channels4 = skip4.get_shape().as_list()[3]
            
            # 4.
            deconv3_fs = self.conv_2d_transpose( skip4, [3,3,128,channels4], conv3_stride.get_shape().as_list(), stride=2, name="deconv3_fs")
            debn3_fs = tf.nn.elu(deconv3_fs)
            
            skip3 = tf.concat([debn3_fs, conv3_stride], 3)
            channels3 = skip3.get_shape().as_list()[3]
            
            # 3.
            deconv2_fs = self.conv_2d_transpose( skip3, [3,3,64,channels3], conv2_stride.get_shape().as_list(), stride=2, name="deconv2_fs")
            debn2_fs = tf.nn.elu(deconv2_fs)
            
            skip2 = tf.concat([debn2_fs, conv2_stride], 3)
            channels2 = skip2.get_shape().as_list()[3]
            
            # 2.
            deconv1_fs = self.conv_2d_transpose( skip2, [3,3,32,channels2], conv1_stride.get_shape().as_list(), stride=2, name="deconv1_fs")
            debn1_fs = tf.nn.elu(deconv1_fs)    
            
            skip1 = tf.concat([debn1_fs, conv1_stride], 3)
            channels1 = skip1.get_shape().as_list()[3]
            
            # 1.
            recon = self.conv_2d_transpose( skip1, [3,3,3,channels1],  images.get_shape().as_list(), stride=2, name="recon") 
        return recon

    def discriminator(self, images, is_train, reuse=None):
        with tf.compat.v1.variable_scope('DIS', reuse=reuse):
            conv1 = self.conv_2d_layer(images, [4,4,3,64], stride=2, name="conv1" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            conv2 = self.conv_2d_layer(bn1, [4,4,64,128], stride=2, name="conv2")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            conv3 = self.conv_2d_layer(bn2, [4,4,128,256], stride=2, name="conv3")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            conv4 = self.conv_2d_layer(bn3, [4,4,256,512], stride=2, name="conv4")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))


            output = self.new_fc_layer( bn4, output_size=1, name='output')

        return output[:,0]       
