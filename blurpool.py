import numpy as np
import keras.backend as K
from keras.layers import Layer


class BlurPool2D(Layer):
    """
    https://arxiv.org/abs/1904.11486 https://github.com/adobe/antialiased-cnns
    """
    def __init__(self, kernel_size=5, stride=2, **kwargs):
        
        self.strides = (stride, stride)
        self.kernel_size = kernel_size
        self.padding = ((int(1 * (kernel_size - 1) / 2), int(np.ceil(1 * (kernel_size - 1) / 2))),
                        (int(1 * (kernel_size - 1) / 2), int(np.ceil(1 * (kernel_size - 1) / 2))))

        if self.kernel_size==1:     self.kernel = [1,]
        elif self.kernel_size==2:   self.kernel = [1, 1]
        elif self.kernel_size==3:   self.kernel = [1, 2, 1]
        elif self.kernel_size==4:   self.kernel = [1, 3, 3, 1]
        elif self.kernel_size==5:   self.kernel = [1, 4, 6, 4, 1]
        elif self.kernel_size==6:   self.kernel = [1, 5, 10, 10, 5, 1]
        elif self.kernel_size==7:   self.kernel = [1, 6, 15, 20, 15, 6, 1]
        self.kernel = np.array(self.kernel, dtype=np.float32)

        super(BlurPool2D, self).__init__(**kwargs)


    def compute_output_shape(self, input_shape):
        height = input_shape[1] // self.strides[0]
        width = input_shape[2] // self.strides[1] 
        channels = input_shape[3]
        return (input_shape[0], height, width, channels)
        

    def call(self, x):
        k = self.a
        k = k[:, None] * k[None, :]
        k = k / K.sum(k)
        k = K.tile (k[:, :, None, None], (1, 1, K.int_shape(x)[-1], 1))                
        k = K.constant(k, dtype=K.floatx())
        x = K.spatial_2d_padding(x, padding=self.padding)
        x = K.depthwise_conv2d(x, k, strides=self.strides, padding='valid')
        return x
