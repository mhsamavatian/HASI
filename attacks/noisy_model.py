import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer,Lambda
from keras import layers
from keras.models import Model
import numpy as np
VGG_MEAN = [103.939, 116.779, 123.68]
class Noise(Layer):
    def __init__(self,**kwargs):
        super(Noise, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Noise, self).build(input_shape)
        
    def call(self, inputs):
        def myfunc(a):
            b,c = a
            
            return b*(1+K.random_uniform(b.shape,minval=-c[0],maxval=c[0]))
        return K.map_fn(myfunc,inputs,dtype="float32")
    
    def compute_output_shape(self, input_shape):
        shape_a, shape_b = input_shape
        return shape_a

def scaling_tf(X, input_range_type):
    """
    Convert to [0, 255], then subtracting means, convert to BGR.
    """

    if input_range_type == 1:
        # The input data range is [0, 1]. 
        # Convert to [0, 255] by multiplying 255
        X = X*255
    elif input_range_type == 2:
        # The input data range is [-0.5, 0.5]. Convert to [0,255] by adding 0.5 element-wise.
        X = (X+0.5)*255
    elif input_range_type == 3:
        # The input data range is [-1, 1]. Convert to [0,1] by x/2+0.5.
        X = (X/2+0.5)*255

    # Caution: Resulting in zero gradients.
    # X_uint8 = tf.clip_by_value(tf.rint(X), 0, 255)
    red, green, blue = tf.split(X, 3, 3)
    X_bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
            # TODO: swap 0 and 2. should be 2,1,0 according to Keras' original code.
        ], 3)

    # x[:, :, :, 0] -= 103.939
    # x[:, :, :, 1] -= 116.779
    # x[:, :, :, 2] -= 123.68
    return X_bgr

def make_new_VGG16_var_n(input_shape=(224, 224, 3),classes=1000,map_=np.zeros(16)):#input_shape=(224* 224* 3+1,)
    i=0
    #inn = layers.Input(shape=input_shape)
    #flat_img = layers.Lambda(lambda x: tf.split(x, [150528,1],1)[0])(inn)
    #img_input = layers.Reshape((224, 224, 3))(flat_img)
    #nois_input = layers.Lambda(lambda x: tf.split(x, [150528,1],1)[1])(inn)
    img_input = layers.Input(shape=input_shape)
    nois_input = layers.Input(shape=(1,),dtype='float32')

    scaler = lambda x: x+0.5
    scaler_layer = Lambda(scaler, input_shape=input_shape)(img_input)

    x = Lambda(lambda x: scaling_tf(x, 1))(scaler_layer)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    
    i+=1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    
    i+=1
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
        
    i+=1
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
        
    i+=1
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
        
    i+=1
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    #if include_top:
        # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(4096, 
                     activation='relu', 
                     name='fc1')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.Dense(4096, 
                     activation='relu', 
                     name='fc2')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    x = layers.Dense(classes, 
                     activation='softmax', 
                     name='predictions')(x)
    if map_[i] == 0:
        x = Noise()([x,nois_input])
    i+=1
    #model = Model(inn, x, name='vgg16')
    model = Model([img_input,nois_input], x, name='vgg16')
    return model
def upload_weights(model,new_model,nios_range=None,nios=True):
    index=0
    i=1
    count=0
    print ('initilize weights ....')
    while i <len(model.layers):
        if model.layers[i].get_config()['name'] == new_model.layers[index].get_config()['name']:
            count+=1
            
            W = model.layers[i].get_weights()
            new_model.layers[index].set_weights(W)  
            index+=1
            i+=1
        else:
            index+=1
            
    print (count, ' layers are initialized\n',10*'*')
    return new_model