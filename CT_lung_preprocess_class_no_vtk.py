# -*- coding: utf-8 -*-
"""
Created on %22-Aug-2020 at 9.43A.m

@author: %A.Nishanth C00294860
"""
import os
import numpy as np
import pydicom
import scipy.ndimage
from skimage import measure, morphology
import pickle
from sklearn.cluster import KMeans
from copy import deepcopy
import matplotlib.pyplot as plt

#from vtk.util import numpy_support
#import vtk
#reader = vtk.vtkDICOMImageReader()

#%%
class CT_lung_preprocess:

    '''
    Important preprocessing steps
    '''
    def __init__(self,master_main_input,my_data_dir,master_main_output):
        
        self.master_main_output=master_main_output
                #assigning input directories
        self.test_input_dir=''.join([master_main_input,'test/'])
        self.models_dir=''.join([my_data_dir,'/model/'])
        
        #preperocessing directories
        #extracted CT_scan portion
        
        self.mask_dir=''.join([master_main_output,'/Test_preprocess/masks/'])
        self.spacing_dir=''.join([master_main_output,'/Test_preprocess/spacing/'])
        self.preprocess_dir=''.join([master_main_output,'/Test_preprocess/rescal_final/'])
        print("CT_lung_process_class_intiated sucessfull")
   
    def  preprocess_all(self):
   
        #creating saving/ output directories 
        os.chdir('/')
        if not os.path.isdir(self.mask_dir):
            os.makedirs(self.mask_dir)
        os.chdir('/')
        if not os.path.isdir(self.spacing_dir):
            os.makedirs(self.spacing_dir)
        os.chdir('/')
        if not os.path.isdir(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        os.chdir('/')
        os.chdir(self.test_input_dir)
        names=os.listdir()
        for name in names:   
            print("Intiating CT extracting from ",name)
            self.save_mask(name)
            print("CT extracted from ",name," sucessfull ")
            
    def save_mask(self,name):
        '''
         Save the mask
        '''

        scan_HU,thikness_record,PixelSpacing= self.load_scan(name)

        Scan_rescaled, spacing = self.rescale(scan_HU, thikness_record,PixelSpacing)
      
        os.chdir('/')
        os.chdir(self.spacing_dir)
        pickle.dump(spacing, open(''.join([name,'_spacing.p']), "wb" ))
        
        change=False
        mask_images=[]
        applied=np.empty((np.size(Scan_rescaled,axis=0),np.size(Scan_rescaled,axis=1),np.size(Scan_rescaled,axis=2)))
        print(np.size(Scan_rescaled,axis=0))
        for i in range(0,np.size(Scan_rescaled,axis=0)):
            # print(i) 
            # self.prob_NAN=Scan_rescaled[i]
            if i==np.size(Scan_rescaled,axis=0)-1:
                try:
                    applied_t,mask_temp=self.segment_lung_mask(Scan_rescaled[i], display=False)
                    applied[i,:,:]=applied_t 
                    mask_images.append(deepcopy(mask_temp))
                except:
                    applied[i,:,:]=0
                    change=True
                    Scan_rescaled= np.delete(Scan_rescaled,-1,0)
                    scan_HU= np.delete(scan_HU,-1,0)
            else:
                applied_t,mask_temp=self.segment_lung_mask(Scan_rescaled[i], display=False)
                applied[i,:,:]=applied_t 
                mask_images.append(deepcopy(mask_temp))
        if change:
            # Scan_rescaled
            applied_new=np.empty((np.size(Scan_rescaled,axis=0),np.size(Scan_rescaled,axis=1),np.size(Scan_rescaled,axis=2)))
            for i in range(0,np.size(Scan_rescaled,axis=0)):
                applied_new[i,:,:]=applied[i,:,:]
            print(name,": Last z removed and Saved")
            
        os.chdir('/')
        os.chdir(self.preprocess_dir)
        np.save(''.join([name,'_rescaled_mask_applied.npy']),applied)
        os.chdir('/')
        os.chdir(self.mask_dir)
        pickle.dump(mask_images, open(''.join([name,'_rescaled_mask.p']), "wb" ))

#    def get_img(self,path_main):
#    
#        array=[]
#        for s in os.listdir(path_main):
#            path=''.join([path_main,'/' + s])
#            reader.SetFileName(path) 
#            reader.SetFileName(path)
#            reader.Update()
#            _extent = reader.GetDataExtent()
#            ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
#    
#            # ConstPixelSpacing = reader.GetPixelSpacing()
#            imageData = reader.GetOutput()
#            pointData = imageData.GetPointData()
#            arrayData = pointData.GetArray(0)
#            ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
#            ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
#    
#            array.append(np.rot90(np.array(ArrayDicom[:,:,0])))
#        image=np.zeros((len(array),np.size(array[0],axis=0),np.size(array[0],axis=1)))
#        for i in range(0,len(array)):
#            image[i,:,:]=array[i]+1024
#        # ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
#        # ArrayDicom = cv2.resize(ArrayDicom,(512,512))
#        return image#,ConstPixelDims,ConstPixelSpacing
           
    def load_scan(self,name):
        """
        Loads scans from a folder and into a list.
        
        Parameters: path (Folder path)
        Converts raw images to Hounsfield Units (HU).
        Returns: image (NumPy array)
        """
        path=''.join([self.test_input_dir,name])

        try:
            slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key = lambda x: int(x.InstanceNumber))
            image = np.stack([s.pixel_array for s in slices])
        except:
            print("Vtk libarry intiated")
            image=self.get_img(path)
        try:
            slice_thickness= np.abs(slices[0].SliceThickness)

        except:
            try:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
                print("slice_thickness-1:",slice_thickness)
            except:
                print("slice_thickness-2:",slice_thickness)
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        print("slice_thickness:",slice_thickness)            
        # for s in slices:
        #     s.SliceThickness = slice_thickness
    
        s=slices[0]
        '''
        Converts raw images to Hounsfield Units (HU).
        
        Convert to Hounsfield units
        
        as per https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/ on 19-Aug-2020 9.38p.m 
        An example: CT images, whose pixel values are measured in Hounsfield units, which can have negative values, 
        are commonly stored with an unsigned integer. As a consequence, it is common for CT DICOM files to have a negative intercept. 
        '''
        
        image = image.astype(np.int16)
        # Convert to Hounsfield units (HU)
        intercept = s.RescaleIntercept
        slope = s.RescaleSlope
        PixelSpacing= s.PixelSpacing#helpfiul for rescaling
        if slope != 1:
            image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
            
        image += np.int16(intercept)
        return np.array(image, dtype=np.float64),slice_thickness,PixelSpacing
    
    
    def rescale(self,image,thikness_record,PixelSpacing,new_spacing=[1,1,1]):
        # Determine current pixel spacing
        spacing = map(float, ([thikness_record] + list(PixelSpacing)))
        spacing = np.array(list(spacing))
        print(spacing)
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        
        return image, new_spacing
    

    '''
    Preprocess for rescling done here
    Start masking
    '''
    
    def segment_lung_mask(self,img,display=False):
        '''   
    
        Parameters
        ----------
        img : numpy array 
         better to have rescaled CT scan image.
        
        Returns
        -------
        Extracted_lung_portion, and mask
        '''
        orig_image=deepcopy(img)
        row_size= img.shape[0]
        col_size = img.shape[1]
        
        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
        mean = np.mean(middle)  
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the 
        # underflow and overflow on the pixel spectrum
        img[img==max]=mean
        img[img==min]=mean
        #
        # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    
        # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
        # We don't want to accidentally clip the lung.
    
        eroded = morphology.erosion(thresh_img,np.ones([3,3]))
        dilation = morphology.dilation(eroded,np.ones([8,8]))
    
        labels = measure.label(dilation) # Different labels are displayed in different colors
        # label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
                good_labels.append(prop.label)
        mask = np.ndarray([row_size,col_size],dtype=np.int8)
        mask[:] = 0
    
        #
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    
        if (display):
            fig, ax = plt.subplots(3, 2, figsize=[12, 12])
            ax[0, 0].set_title("Original")
            ax[0, 0].imshow(img, cmap='gray')
            ax[0, 0].axis('off')
            ax[0, 1].set_title("Threshold")
            ax[0, 1].imshow(thresh_img, cmap='gray')
            ax[0, 1].axis('off')
            ax[1, 0].set_title("After Erosion and Dilation")
            ax[1, 0].imshow(dilation, cmap='gray')
            ax[1, 0].axis('off')
            ax[1, 1].set_title("Color Labels")
            ax[1, 1].imshow(labels)
            ax[1, 1].axis('off')
            ax[2, 0].set_title("Final Mask")
            ax[2, 0].imshow(mask, cmap='gray')
            ax[2, 0].axis('off')
            ax[2, 1].set_title("Apply Mask on Original")
            ax[2, 1].imshow(mask*img, cmap='gray')
            ax[2, 1].axis('off')
            
            plt.show()
        return mask*orig_image,mask
