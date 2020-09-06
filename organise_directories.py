# -*- coding: utf-8 -*-
"""
Created on %06-Sep-2020 at 2.29 p.m

@author: %A.Nishanth C00294860

This script is specially  written for creating and organising the directories needed in the OSIC compection

"""
import os

#master_main_input='/kaggle/input/osic-pulmonary-fibrosis-progression/'
#master_main_output= '/kaggle/working/'
#my_data_dir='kaggle/input/my-files/Lung_OSIC_kaggle-master/'


master_main_input='G:/Keggle_project_Fall_2020/'
master_main_output='G:/Keggle_project_Fall_2020/'
my_data_dir='G:/Keggle_project_Fall_2020/GitHub/'
'''
#assigning input directories
test_input=''.join([master_main_input,'test/'])
models_dir=''.join([my_data_dir,'model/'])

#preperocessing directories
#extracted CT_scan portion

mask_dir=''.join([master_main_output,'/Test_preprocess/masks/'])
spacing_dir=''.join([master_main_output,'/Test_preprocess/spacing/'])
preprocess_dir=''.join([master_main_output,'/Test_preprocess/rescal_final/'])

#creating saving/ output directories 
os.chdir('/')
if not os.path.isdir(mask_dir):
    os.mkdir(mask_dir)
os.chdir('/')
if not os.path.isdir(spacing_dir):
    os.mkdir(spacing_dir)
os.chdir('/')
if not os.path.isdir(preprocess_dir):
    os.mkdir(preprocess_dir)

'''