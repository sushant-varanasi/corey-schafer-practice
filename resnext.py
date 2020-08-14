#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.layers import Activation
from keras.layers import Add
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.datasets import cifar10
import numpy as np


# In[2]:


batch_size = 32
epochs = 200
num_classes = 10


# In[3]:


#Loading the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Data Normalization
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#Subtracting mean pixel value
x_train_mean = np.mean(x_train, axis = 0)
x_train -= x_train_mean
x_test -= x_train_mean

input_shape = x_train.shape[1:]

#Changing format of label data to use categorical cross entropy loss
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[4]:


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


# In[5]:


def split(inputs, cardinality):
    inputs_channels = inputs.shape[3]
    group_size = inputs_channels // cardinality    
    groups = list()
    for number in range(1, cardinality+1):
        begin = int((number-1)*group_size)
        end = int(number*group_size)
        block = Lambda(lambda x:x[:,:,:,begin:end])(inputs)
        groups.append(block)
    return groups


# In[6]:


def transform(groups, filters, strides, stage, block):
    f1, f2 = filters    
    conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage), block=str(block))
    bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage), block=str(block))
    
    transformed_tensor = list()
    i = 1
    
    for inputs in groups:
        # first conv of the transformation phase
        x = Conv2D(filters=f1, kernel_size=(1,1), strides=strides, padding="valid", 
                   name=conv_name+'1a_split'+str(i), kernel_initializer=glorot_uniform(seed=0))(inputs)
        x = BatchNormalization(axis=3, name=bn_name+'1a_split'+str(i))(x)
        x = Activation('relu')(x)

        # second conv of the transformation phase
        x = Conv2D(filters=f2, kernel_size=(3,3), strides=(1,1), padding="same", 
                   name=conv_name+'1b_split'+str(i), kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis=3, name=bn_name+'1b_split'+str(i))(x)
        x = Activation('relu')(x)
        
        # Add x to transformed tensor list
        transformed_tensor.append(x)
        i+=1
        
    # Concatenate all tensor from each group
    x = Concatenate(name='concat'+str(stage)+''+block)(transformed_tensor)
    
    return x


# In[7]:


def transition(inputs, filters, stage, block):
    x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), padding="valid", 
                   name='conv2d-trans'+str(stage)+''+block, kernel_initializer=glorot_uniform(seed=0))(inputs)
    x = BatchNormalization(axis=3, name='batchnorm-trans'+str(stage)+''+block)(x)
    x = Activation('relu')(x)
    
    return x


# In[8]:


def identity_block(inputs, filters, cardinality, stage, block, strides=(1,1)):
    
    conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage),block=str(block))
    bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage),block=str(block))
    
    #save the input tensor value
    x_shortcut = inputs
    x = inputs
    
    f1, f2, f3 = filters
    
    # divide input channels into groups. The number of groups is define by cardinality param
    groups = split(inputs=x, cardinality=cardinality)
    
    # transform each group by doing a set of convolutions and concat the results
    f1 = int(f1 / cardinality)
    f2 = int(f2 / cardinality)
    x = transform(groups=groups, filters=(f1, f2), strides=strides, stage=stage, block=block)
    
    # make a transition by doing 1x1 conv
    x = transition(inputs=x, filters=f3, stage=stage, block=block)
    
    # Last step of the identity block, shortcut concatenation
    x = Add()([x,x_shortcut])
    x = Activation('relu')(x)
    
    return x


# In[9]:


def downsampling(inputs, filters, cardinality, strides, stage, block):
    
    # useful variables
    conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage), block=str(block))
    bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage), block=str(block))
    
    # Retrieve filters for each layer
    f1, f2, f3 = filters
    
    # save the input tensor value
    x_shortcut = inputs
    x = inputs
    
    # divide input channels into groups. The number of groups is define by cardinality param
    groups = split(inputs=x, cardinality=cardinality)
    
    # transform each group by doing a set of convolutions and concat the results
    f1 = int(f1 / cardinality)
    f2 = int(f2 / cardinality)
    x = transform(groups=groups, filters=(f1, f2), strides=strides, stage=stage, block=block)
    
    # make a transition by doing 1x1 conv
    x = transition(inputs=x, filters=f3, stage=stage, block=block)
    
    # Projection Shortcut to match dimensions 
    x_shortcut = Conv2D(filters=f3, kernel_size=(1,1), strides=strides, padding="valid", 
               name='{base}2'.format(base=conv_name), kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
    x_shortcut = BatchNormalization(axis=3, name='{base}2'.format(base=bn_name))(x_shortcut)
    
    # Add x and x_shortcut
    x = Add()([x,x_shortcut])
    x = Activation('relu')(x)
    
    return x


# In[10]:


def ResNeXt50(input_shape, classes):
    
    # Transform input to a tensor of shape input_shape 
    x_input = Input(input_shape)
    
    # Add zero padding
    x = ZeroPadding2D((3,3))(x_input)
    
    # Initial Stage. Let's say stage 1
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), 
               name='conv2d_1', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name='batchnorm_1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)
    
    # Stage 2
    x = downsampling(inputs=x, filters=(128,128,256), cardinality=4, strides=(2,2), stage=2, block="a")
    x = identity_block(inputs=x, filters=(128,128,256), cardinality=4, stage=2, block="b")
    x = identity_block(inputs=x, filters=(128,128,256), cardinality=4, stage=2, block="c")
    
    
    # Stage 3
    x = downsampling(inputs=x, filters=(256,256,512), cardinality=4, strides=(2,2), stage=3, block="a")
    x = identity_block(inputs=x, filters=(256,256,512), cardinality=4, stage=3, block="b")
    x = identity_block(inputs=x, filters=(256,256,512), cardinality=4, stage=3, block="c")
    x = identity_block(inputs=x, filters=(256,256,512), cardinality=4, stage=3, block="d")
    
    
    # Stage 4
    x = downsampling(inputs=x, filters=(512,512,1024), cardinality=4, strides=(2,2), stage=4, block="a")
    x = identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="b")
    x = identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="c")
    x = identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="d")
    x = identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="e")
    x = identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="f")
    
    
    # Stage 5
    x = downsampling(inputs=x, filters=(1024,1024,2048), cardinality=4, strides=(2,2), stage=5, block="a")
    x = identity_block(inputs=x, filters=(1024,1024,2048), cardinality=4, stage=5, block="b")
    x = identity_block(inputs=x, filters=(1024,1024,2048), cardinality=4, stage=5, block="c")
    
    
    # Average pooling
    x = AveragePooling2D(pool_size=(2,2), padding="same")(x)
    
    # Output layer
    x = Flatten()(x)
    x = Dense(classes, activation="softmax", kernel_initializer=glorot_uniform(seed=0), 
              name="fc{cls}".format(cls=str(classes)))(x)
    
    # Create the model
    model = Model(inputs=x_input, outputs=x, name="resnet50")
    
    return model


# In[15]:


model = ResNeXt50(input_shape=(32,32,3), classes=10)

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# In[16]:


# Prepare model model saving directory.
import os
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


# In[17]:


# Prepare callbacks for model saving and for learning rate adjustment.
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]


# In[ ]:


model.fit(x_train[:10000], y_train[:10000],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test[:2500], y_test[:2500]),
              shuffle=True,
              callbacks=callbacks)


# In[ ]:




