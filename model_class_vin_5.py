# -*- coding: utf-8 -*-
"""
Created on %14-Sep-2020 at 9.48 A.m

@author: c00294860

This code is formalised as class build earlier
"""
from tensorflow import keras
from tensorflow.keras import layers


'''Defining the swish -activation function'''
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation


class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'
        self.supports_masking = True

def swish(x, beta = 1):
    '''
    "Swish:

    In 2017, Researchers at Google Brain, published a new paper where they proposed their novel activation function named as Swish.
     Swish is a gated version of Sigmoid Activation Function, where the mathematical formula is: f(x, β) = x * sigmoid(x, β) 
     [Often Swish is referred as SILU or simply Swish-1 when β = 1].

    Their results showcased a significant improvement in Top-1 Accuracy for ImageNet at 0.9% better than ReLU for Mobile 
    NASNet-A and 0.7% for Inception ResNet v2." from https://towardsdatascience.com/mish-8283934a72df on  Fri Dec 27 14:44:30 2019
    
    '''
    return (x * sigmoid(beta * x))

get_custom_objects().clear()
get_custom_objects().update({'swish': Swish(swish)})   
 
class model_with_fixed_size:
    
    def __init__(self,activation='relu',square_height=512,square_width=512,hidden_dence_size=100,h_size_stack=128):

        self.activation = activation
        #CNN feed image parameters
        self.square_height=square_height
        self.square_width=square_width
        #LSTM parameters
        self.hidden_dence_size=hidden_dence_size
        self.h_size_stack=h_size_stack

    def model_maker(self,p1=3,d1=32):

        k1=p1
        k2=p1
        k3=p1
        k4=p1
        print("k1: ",k1)
        inputs = keras.Input(shape=(None,self.square_height, self.square_width,1))
        meta_data_input=keras.Input(shape=(4,))
        
        xy = layers.TimeDistributed(layers.Conv2D(d1, (k1,k1),strides=(1,1),padding='same',activation= self.activation,name='conv_layer_1'))(inputs)
        block_1_xy_output =  layers.TimeDistributed(layers.MaxPooling2D(pool_size=(4, 4)))(xy)
        
        xy = layers.TimeDistributed(layers.Conv2D(d1, (k2,k2),strides=(1,1),padding='same',activation= self.activation,name='conv_layer_2'))(block_1_xy_output)
        block_2_xy_output = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(xy)
        
        xy = layers.TimeDistributed(layers.Conv2D(2*d1, (k3,k3), padding='same',activation= self.activation,name='conv_layer_3'))(block_2_xy_output)
        xy = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(xy)
        
        xy = layers.TimeDistributed(layers.Conv2D(2*d1, (k4,k4), padding='same',activation= self.activation,name='conv_layer_4'))(xy)
        block_3_xy_output = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(xy)
        #
        xy = layers.TimeDistributed(layers.Conv2D(4*d1, (k4,k4), padding='same',activation= self.activation,name='conv_layer_5'))(block_3_xy_output)
        block_4_xy_output_b = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(xy)
        block_4_xy_output=layers.TimeDistributed(layers.Flatten())(block_4_xy_output_b)
        model_name_CNN =''.join(['model_fixed_size_CNN_p1_',str(p1),'_d1_',str(d1),'.h5'])

        cnn_ext_fea_model = keras.models.Model(inputs=inputs,outputs= block_4_xy_output, name=model_name_CNN)     
        #cnn_ext_fea_model.summary()
        
        cnn_ext_fea=cnn_ext_fea_model(inputs)
        
#        lstm_1 = layers.LSTM(self.h_size_stack)(block_4_xy_output)
        lstm_1 = layers.LSTM(self.h_size_stack)(cnn_ext_fea)
        Hiden_dense =layers.Dense(self.hidden_dence_size)(lstm_1)
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)
        merged = layers.concatenate([DROP_OUT, meta_data_input], axis=1)
        outputs_1 = layers.Dense(1,activation=self.activation)(merged)
        gradient_model = keras.models.Model(inputs=[inputs,meta_data_input], outputs=outputs_1, name='gradient_1')     
        outputs_3= gradient_model(inputs=[inputs,meta_data_input])     

        lstm_2 = layers.LSTM(self.h_size_stack)(cnn_ext_fea)
        Hiden_dense_2 =layers.Dense(self.hidden_dence_size)(lstm_2)
        DROP_OUT_2= layers.Dropout(0.5)(Hiden_dense_2)
        merged_2 = layers.concatenate([DROP_OUT_2, meta_data_input], axis=1)
        outputs_2 = layers.Dense(1,activation=self.activation)(merged_2)
        cross_sec_model = keras.models.Model(inputs=[inputs,meta_data_input], outputs=outputs_2, name='cross_sec_model')     
        outputs_4= cross_sec_model(inputs=[inputs,meta_data_input])     

        merged = layers.concatenate([outputs_3,outputs_4], axis=1)
        output_merged = layers.Flatten()(merged)

        model_name =''.join(['model_fixed_size_CNN_p1_',str(p1),'_d1_',str(d1),'_LSTM_nodes_',str(self.h_size_stack),'.h5'])
        model = keras.models.Model(inputs=[inputs,meta_data_input], outputs=output_merged,name=model_name)
        model.summary()
        return model,model_name

class model_only_LSTM_portion:
    
    def __init__(self,activation='relu',hidden_dence_size=100,h_size_stack=128):

        self.activation = activation
        #LSTM parameters
        self.hidden_dence_size=hidden_dence_size
        self.h_size_stack=h_size_stack

    def model_maker(self,model_name,d1=3072):

        inputs = keras.Input(shape=(None,d1))
        meta_data_input=keras.Input(shape=(4,))
        
#        lstm_1 = layers.LSTM(self.h_size_stack)(block_4_xy_output)
        lstm_1 = layers.LSTM(self.h_size_stack)(inputs)
        Hiden_dense =layers.Dense(self.hidden_dence_size)(lstm_1)
        merged = layers.concatenate([Hiden_dense, meta_data_input], axis=1)
        outputs_1 = layers.Dense(1,activation=self.activation)(merged)
        LSTM_model = keras.models.Model(inputs=[inputs,meta_data_input], outputs=outputs_1, name=model_name)     
        LSTM_model.summary()
        return LSTM_model

class model_only_LSTM_portion_with_dropout:
    
    def __init__(self,activation='relu',hidden_dence_size=100,h_size_stack=128):

        self.activation = activation
        #LSTM parameters
        self.hidden_dence_size=hidden_dence_size
        self.h_size_stack=h_size_stack

    def model_maker(self,model_name,d1=3072,drop_out=0.5):

        inputs = keras.Input(shape=(None,d1))
        meta_data_input=keras.Input(shape=(4,))
        
#        lstm_1 = layers.LSTM(self.h_size_stack)(block_4_xy_output)
        lstm_1 = layers.LSTM(self.h_size_stack)(inputs)
        Hiden_dense =layers.Dense(self.hidden_dence_size)(lstm_1)
        DROP_OUT= layers.Dropout(0.5)(Hiden_dense)
        merged = layers.concatenate([DROP_OUT, meta_data_input], axis=1)
#        merged = layers.concatenate([Hiden_dense, meta_data_input], axis=1)
        outputs_1 = layers.Dense(1,activation=self.activation)(merged)
        LSTM_model = keras.models.Model(inputs=[inputs,meta_data_input], outputs=outputs_1, name=model_name)     
        LSTM_model.summary()
        return LSTM_model

class model_visual_extractor_portion_fixed_size:
    
    def __init__(self,activation='relu',square_height=512,square_width=512):

        self.activation = activation
        #CNN feed image parameters
        self.square_height=square_height
        self.square_width=square_width


    def model_maker(self,p1=3,d1=32):

        k1=p1
        k2=p1
        k3=p1
        k4=p1
        print("k1: ",k1)
        inputs = keras.Input(shape=(None,self.square_height, self.square_width,1))
        
        xy = layers.TimeDistributed(layers.Conv2D(d1, (k1,k1),strides=(1,1),padding='same',activation= self.activation,name='conv_layer_1'))(inputs)
        block_1_xy_output =  layers.TimeDistributed(layers.MaxPooling2D(pool_size=(4, 4)))(xy)
        
        xy = layers.TimeDistributed(layers.Conv2D(d1, (k2,k2),strides=(1,1),padding='same',activation= self.activation,name='conv_layer_2'))(block_1_xy_output)
        block_2_xy_output = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(xy)
        
        xy = layers.TimeDistributed(layers.Conv2D(2*d1, (k3,k3), padding='same',activation= self.activation,name='conv_layer_3'))(block_2_xy_output)
        xy = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(xy)
        
        xy = layers.TimeDistributed(layers.Conv2D(2*d1, (k4,k4), padding='same',activation= self.activation,name='conv_layer_4'))(xy)
        block_3_xy_output = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(xy)
        #
        xy = layers.TimeDistributed(layers.Conv2D(4*d1, (k4,k4), padding='same',activation= self.activation,name='conv_layer_5'))(block_3_xy_output)
        block_4_xy_output_b = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(xy)
        block_4_xy_output=layers.TimeDistributed(layers.Flatten())(block_4_xy_output_b)
        model_name_CNN =''.join(['model_fixed_size_CNN_p1_',str(p1),'_d1_',str(d1),'.h5'])

        model = keras.models.Model(inputs=inputs,outputs= block_4_xy_output, name=model_name_CNN)     
        return model