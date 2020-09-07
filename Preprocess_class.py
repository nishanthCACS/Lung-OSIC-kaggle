#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10.00Am 2020

@author: c00294860
"""
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt  

class preprocess:
    
    def __init__(self,loading_CNN_dir,loading_mask_dir,loading_preprocess,load_spacing_directory,saving_dir_main,sving_image_direct='optional'):

        self.loading_CNN_dir = loading_CNN_dir
        self.loading_mask_dir = loading_mask_dir
        self.loading_preprocess=loading_preprocess
        self.load_spacing_directory=load_spacing_directory

        self.sving_image_direct=sving_image_direct

        self.saving_dir=''.join([saving_dir_main,'/Test_preprocess/extract_rmd_check/'])
        os.chdir('/')
        if not os.path.isdir(self.saving_dir):
            os.makedirs(self.saving_dir)
    
    def feature_extract_all(self):
        '''
        Loading the models and extract the important features
        '''
        print("model load intiated")

        os.chdir('/')
        os.chdir(self.loading_CNN_dir)
        encoder_model = load_model('Feature_extractor_conv_vin_1.h5')
        print("Model load suceeded")
        os.chdir('/')
        os.chdir(self.loading_preprocess)
        names=os.listdir()
        for name in names:
            os.chdir('/')
            os.chdir(self.loading_preprocess)
            chk=np.load(name)
            print(name," intiated")
            chk=chk.reshape(chk.shape[0],chk.shape[1],chk.shape[2],1)
            self.predicted=encoder_model.predict(chk,verbose=0)
            
            os.chdir('/')
            os.chdir(self.load_spacing_directory)
            self.space=pickle.load(open(''.join([name[0:25],'_spacing.p']),'rb'))
            self.feature_extract_main(name)
            print(name," sucessfull")

    def plot_important_feature(self,sel_image):
        
        os.chdir('/')
        os.chdir(self.sving_image_direct)
        m=0
        for i in range(0,64,16):
            fig, axs = plt.subplots(4, 4)
            for j in range(0,4):
                for k in range(0,4):
                    axs[j, k].imshow(self.predicted[sel_image,:,:,m])
                    m=m+1
            fig.savefig(''.join(['features_',str(i),'_to_',str(m),"_changed.png"]))
  
    def feature_extract_main(self,name):    
       '''
       The features are selected based on visual inspection
       selected_features=[0,3,11,15,17,18,29,30,41,51,52,53,54,55,58,63]
       '''
       print("Feature extract main intiated")

       feature_extracted_final=np.empty((16,4))
    
       self.selected_features=[0,3,11,15,17,18,29,30,41,51,52,53,54,55,58,63]

       for m in range(0,len(self.selected_features)):
           volume_final,area_final,change_area_positive,change_area_negative = self.given_feature_extract(m,name)
           feature_extracted_final[m,0]=volume_final
           feature_extracted_final[m,1]=area_final
           feature_extracted_final[m,2]=change_area_positive
           feature_extracted_final[m,3]=change_area_negative
           
       os.chdir('/')
       os.chdir(self.saving_dir)
       np.save(''.join([name[0:25],'_extract_features.npy']),feature_extracted_final)
       print("done")
       
    def given_feature_extract(self,m,name):    
        
        selected_features=self.selected_features
        os.chdir('/')
        os.chdir(self.loading_mask_dir)
        mask=pickle.load(open(''.join([name[0:25],'_rescaled_mask.p']),"rb"))
       
        # factors
        volume_factor=np.round(self.space[0]*self.space[1]*self.space[2],decimals=2)
        area_factor=np.round(self.space[1]*self.space[2],decimals=2)
        
        area_all=0
        volume=0
        change_area_positive=0
        change_area_negative=0
        for imag_num in range(0,np.size(self.predicted,axis=0)):
           sel_feat=np.round(mask[imag_num]*self.predicted[imag_num,:,:,selected_features[m]],decimals=2)
           area_loc=np.sum(sel_feat)
           area_all=area_all+area_loc
           if imag_num<np.size(self.predicted,axis=0)-1:
               mask_sel=mask[imag_num+1]-mask[imag_num]
               sel_feat_change=self.predicted[imag_num+1,:,:,selected_features[m]]-self.predicted[imag_num,:,:,selected_features[m]]
               
               sel_feat_volume_cal_part_1=0.5*np.sum(np.abs(np.round(mask_sel*(sel_feat_change),decimals=2)))
               
               change_area_positive=change_area_positive+np.sum(np.array(sel_feat_change) >= 0)
               change_area_negative=change_area_negative+np.sum(np.array(sel_feat_change) < 0)
               
               volume=volume+np.abs(sel_feat_volume_cal_part_1)+area_loc
        
        volume_final=volume_factor*volume
        area_final=area_factor*area_all
        
        change_area_positive=area_factor*change_area_positive
        change_area_negative=area_factor*change_area_negative
           
        return volume_final,area_final,change_area_positive,change_area_negative