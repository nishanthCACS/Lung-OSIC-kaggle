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

from vtk.util import numpy_support
import vtk
reader = vtk.vtkDICOMImageReader()

#%%
class CT_lung_preprocess:

    '''
    Important preprocessing steps
    '''
    
    def __init__(self,name,working_dir,saving_dir):
        
        self.working_dir = working_dir
        self.saving_dir = saving_dir
        self.name = name #name of the class
        
    def save_mask(self,only_extracted_final_preprocess_save=True):
        '''
         Save the mask
        '''
        
        # try: 
        #     os.chdir('/')
        #     os.chdir(self.saving_dir)         
        #     Scan_rescaled = pickle.load(open(''.join([self.name,'_scan_rescaled.p']), "rb" ))
            
        # except:
        # first load the files needed
        os.chdir('/')
        os.chdir(self.working_dir)
        scan_HU,thikness_record,PixelSpacing= self.load_scan()
        # thikness_records.append(thikness_record_t)
        Scan_rescaled, spacing = self.rescale(scan_HU, thikness_record,PixelSpacing)
        #check the scaling is that is rgular CT scan or CT scan in box
        if Scan_rescaled.shape[1]>512 or  Scan_rescaled.shape[2]>512:
            Scan_rescaled=self.needed_lung_scan_edition(Scan_rescaled)
#        os.chdir('/')
#        os.chdir(self.saving_dir)
#        pickle.dump(Scan_rescaled, open(''.join([self.name,'_scan_rescaled.p']), "wb" ))
#        pickle.dump(scan_HU, open(''.join([self.name,'_scan_HU.p']), "wb" ))
#        pickle.dump(spacing, open(''.join([self.name,'_spacing.p']), "wb" ))
        
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
#                    scan_HU = pickle.load(open(''.join([self.name,'_scan_HU.p']), "rb" ))
                    scan_HU= np.delete(scan_HU,-1,0)

            else:
                applied_t,mask_temp=self.segment_lung_mask(Scan_rescaled[i],display=False)
                applied[i,:,:]=applied_t 
                mask_images.append(deepcopy(mask_temp))
        if change:
            # Scan_rescaled
            applied_new=np.empty((np.size(Scan_rescaled,axis=0),np.size(Scan_rescaled,axis=1),np.size(Scan_rescaled,axis=2)))
            for i in range(0,np.size(Scan_rescaled,axis=0)):
                applied_new[i,:,:]=applied[i,:,:]
            if not only_extracted_final_preprocess_save:
                pickle.dump(Scan_rescaled, open(''.join([self.name,'_scan_rescaled.p']), "wb" ))
                pickle.dump(scan_HU, open(''.join([self.name,'_scan_HU.p']), "wb" ))
                np.save(''.join([self.name,'_rescaled_mask_applied.npy']),applied_new)
                print(self.name,": Last z removed and Saved")
                pickle.dump(mask_images, open(''.join([self.name,'_rescaled_mask.p']), "wb" ))

        else:
            if not only_extracted_final_preprocess_save:
                os.chdir('/')
                os.chdir(self.saving_dir)
                pickle.dump(mask_images, open(''.join([self.name,'_rescaled_mask.p']), "wb" ))
                np.save(''.join([self.name,'_rescaled_mask_applied.npy']),applied)
                applied_new=applied
                
        self.extract_final_CNN_feature(applied_new,mask_images)
        
    def extract_final_CNN_feature(self,applied_new,mask_images,return_only=False):
        '''
        This function         
        '''
#        feat_ext_final=np.empty((np.size(applied_new,axis=0),np.size(applied_new,axis=1),np.size(applied_new,axis=2)))
        feat_ext_final=np.empty((np.size(applied_new,axis=0),512,512))

        for i in range(0,np.size(applied_new,axis=0)):
            mask = mask_images[i]
            image= applied_new[i]
            feat_ext_final[i,:,:]=deepcopy(self.change_scales_back(image,mask))
        if not return_only:
            os.chdir('/')
            os.chdir(self.saving_dir)
            np.save(''.join([self.name,'_CNN_extracted.npy']),feat_ext_final)
        else:
            return feat_ext_final
                
    def get_img(self,path_main):
    
        array=[]
        for s in os.listdir(path_main):
            path=''.join([path_main,'/' + s])
            reader.SetFileName(path) 
            reader.SetFileName(path)
            reader.Update()
            _extent = reader.GetDataExtent()
            ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]
    
            # ConstPixelSpacing = reader.GetPixelSpacing()
            imageData = reader.GetOutput()
            pointData = imageData.GetPointData()
            arrayData = pointData.GetArray(0)
            ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
            ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    
            array.append(np.rot90(np.array(ArrayDicom[:,:,0])))
        image=np.zeros((len(array),np.size(array[0],axis=0),np.size(array[0],axis=1)))
        for i in range(0,len(array)):
            image[i,:,:]=array[i]+1024
        # ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
        # ArrayDicom = cv2.resize(ArrayDicom,(512,512))
        return image#,ConstPixelDims,ConstPixelSpacing
            
    def load_scan(self):
        """
        Loads scans from a folder and into a list.
        
        Parameters: path (Folder path)
        Converts raw images to Hounsfield Units (HU).
        Returns: image (NumPy array)
        """
        path=''.join([self.working_dir,self.name])

        try:
            slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key = lambda x: int(x.InstanceNumber))
            image = np.stack([s.pixel_array for s in slices])
        except:
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
    
    def needed_lung_scan_edition(self,scan_rescaled,display=False):
        '''
        this fucntion crop the CT scan to the wanted portion
        
        '''
        image=scan_rescaled[int(len(scan_rescaled)/2)]
        if display:
            plt.imshow(image)
    
        i=0
        while np.abs(image[i][int(image.shape[1]/2)])<700:
            i=i+1
            break_i=i
        j=0
        while np.abs(image[int(image.shape[0]/2)][j])<700:
            j=j+1
            break_j=j
            
        i=0
        while np.abs(image[i][int(image.shape[1]/2)])<700:
            i=i-1
            break_ot_i=i
        j=0
        while np.abs(image[int(image.shape[0]/2)][j])<700:
            j=j-1
            break_ot_j=j
    
        crop_img = image[break_i:break_ot_i, break_j:break_ot_j]
        cropped_scan = np.empty((len(scan_rescaled),crop_img.shape[0],crop_img.shape[1]))
    #    cropp_image_mask_applied,mask,dilation=segment_lung_mask(crop_img.astype(int))
    #    crop_img = crop_img.astype(int)
    #    plt.imshow(crop_img.astype(int))
        for i in range(0,len(scan_rescaled)):
            image=scan_rescaled[i]
            crop_img = image[break_i:break_ot_i, break_j:break_ot_j]
            cropped_scan[i]=deepcopy(crop_img)
        return cropped_scan
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
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        if self.chk_mask_dialtion_worked(mask):
           mask= self.get_mask_from_dilation(dilation,display=display)

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
    
    def get_mask_from_dilation(self,dilation,display=False):
        '''
        This function retrieve mask from dilation
        '''
        
        dialation_copy=deepcopy(dilation)
        for j in range(0,dilation.shape[1]):
            i=0
            while dilation[i][j]==1 and dilation.shape[0]-1>i:
                dilation[i][j]=0
                i=i+1
        
        for j in range(0,dilation.shape[1]):
            i=-1
            j=j*-1
            while dialation_copy[i][j]==1 and -dilation.shape[0]+1<i:
                dialation_copy[i][j]=0
                i=i-1
            if i>-2:
                i=i-1
                while dialation_copy[i][j]==1 and -dilation.shape[0]+1<i:
                    dialation_copy[i][j]=0
                    i=i-1
            if i>-3:
                i=i-1
                while dialation_copy[i][j]==1 and -dilation.shape[0]+1<i:
                    dialation_copy[i][j]=0
                    i=i-1
        if display:
            plt.imshow(dilation*dialation_copy)
        mask=dilation*dialation_copy
        return mask
    
    def chk_mask_dialtion_worked(self,mask):
        '''
        This fuction will check the mask is created effectively
        '''
        if len(mask)>400:
           mask_loaded= mask[250]
           if np.sum(mask_loaded)>10000:
               mask_recheck=False
           else:
               mask_recheck=True
        elif len(mask)>250:
           mask_loaded= mask[95]
           if np.sum(mask_loaded)>10000:
               mask_recheck=False
           else:
               mask_recheck=True
        elif len(mask)>200:
           mask_loaded= mask[85]
           if np.sum(mask_loaded)>10000:
               mask_recheck=False
           else:
               mask_recheck=True
        elif len(mask)>150:
           mask_loaded= mask[75]
           if np.sum(mask_loaded)>10000:
               mask_recheck=False
           else:
               mask_recheck=True
        elif len(mask)>100:
           mask_loaded= mask[55]
           if np.sum(mask_loaded)>10000:
               mask_recheck=False
           else:
               mask_recheck=True
        elif len(mask)>50:
           mask_loaded= mask[25]
           if np.sum(mask_loaded)>10000:
               mask_recheck=False
           else:
               mask_recheck=True
        elif len(mask)>35:
           mask_loaded= mask[15]
           if np.sum(mask_loaded)>10000:
               mask_recheck=False
           else:
               mask_recheck=True
        else:
           mask_loaded= mask[10]
           if np.sum(mask_loaded)>10000:
               mask_recheck=False
           else:
               mask_recheck=True
        return mask_recheck
    
    def change_scales_back(self,image,mask,air_HU_thresh=-1250,bone_HU_thresh=500):
        '''
        Remove the reducndancy thrugh rescaling the HU 
        This will finalised the scan image set for CNN model
        '''
        for i in range(0,image.shape[0]):
            for j in range(0,image.shape[1]):
                if image[i,j]<air_HU_thresh:
                    image[i,j]=air_HU_thresh
                elif image[i,j]>bone_HU_thresh:
                    image[i,j]=bone_HU_thresh
        
        mask_sel=deepcopy(mask)
        mask_one=np.ones((mask_sel.shape[0],mask_sel.shape[1]))
        mask_change_back=(mask_one-mask_sel)*500
        image=image+mask_change_back
        image=(image+1250)/1750
        
        #fixing the image size to 512 x512
        image_fixed=np.ones((512,512))
        image_fixed[256-int(image.shape[0]/2):256-int(image.shape[0]/2)+image.shape[0],256-int(image.shape[1]/2):256-int(image.shape[1]/2)+image.shape[1]] = image
        image_final=np.ones((512,512))
        image_final=image_final-image_fixed#to make the air near to one and treat outside of the Lung as bone
        return image_final