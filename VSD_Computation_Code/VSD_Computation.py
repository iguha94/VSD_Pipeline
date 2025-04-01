import SimpleITK
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.filters import threshold_local,threshold_li, threshold_otsu, gaussian
from Estimate_Skeleton_radius_3D import *
from Get_Skeleton_info import *
from Propagate_Vessel_ID import *
from scipy import ndimage
import skimage.measure
from PIL import Image, ImageFilter
import scipy.ndimage as ndimage
import os

import numpy as np 
import cv2
from skimage.filters import frangi

basepath='../Sample_Data/'
outpath='../Sample_Data/'

def gaussian(x,mu,sigma):
    return np.exp(-((x-mu)*(x-mu))/(2*sigma*sigma))

#Function to elevate the structures in a microscopic image
def Vesselness_sensitive_CE_3D(pix,kernel_size,cliplimit,grid_size,apply_CLAHE):
    size = pix.shape
    print(size)
    Xdim = size[0]
    Ydim = size[1]
    Zdim = size[2]
    elevated_pix = np.zeros(size)
    vessel_pix = np.zeros(size)
    global_max = 2000
    print('Maximum Intensity: ',global_max)
    vesselness=frangi(pix,sigmas=range(1, 5, 1),black_ridges=False)
    for z in range(0,Xdim):
        slice = pix[z]
        for x in range(0,Ydim):
            for y in range(0,Zdim):
                if slice[x][y]==0:
                    weight=0
                elif slice[x][y]>=global_max:
                    weight=1
                else:
                    weight=1+np.log(global_max/slice[x][y])
                weight=1
                elevated_pix[z][x][y]=int(weight*slice[x][y]*(1.0-gaussian(slice[x][y],170,60.0))*(1-vesselness[z][x][y])) ## Assume that noise is nonzero mean gaussian

    elevated_pix = np.uint16(elevated_pix)
    pix = np.uint16(pix)
    vesselness2 = np.uint16(vesselness*100)
    print('maximum Intensity After: ',np.max(elevated_pix))
    
    if apply_CLAHE:
        print('Applying CLAHE')
        for z in range(kernel_size,Xdim-kernel_size):
            slice = elevated_pix[z]
            #slice = pix[z]
            clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(grid_size,grid_size))
            elevated_pix[z]=cv2.medianBlur(clahe.apply(slice),3)

    return elevated_pix,vesselness2


def GetVesselSizeDistribution(labelskelname,radiusname):
    curveskelimg = SimpleITK.ReadImage(labelskelname)
    curveskelarr = SimpleITK.GetArrayFromImage(curveskelimg)

    curveimg2=SimpleITK.GetImageFromArray(curveskelarr)
    SimpleITK.WriteImage(curveimg2,labelskelname)


    radiusimg = SimpleITK.ReadImage(radiusname)
    radius_arr = SimpleITK.GetArrayFromImage(radiusimg)

    size = curveskelarr.shape
    curve_rad_arr = np.zeros(size)

    totalBranches = int(np.max(curveskelarr))
    #print('Total branches: ',totalBranches)

    radius_dist_arr = np.zeros((totalBranches+1,))
    count_arr = np.zeros((totalBranches+1,))
    count_arr[0]=1
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if curveskelarr[i][j][k]<=0:
                    continue
                branchID = int(curveskelarr[i][j][k])
                radius = radius_arr[i][j][k]
                radius_dist_arr[branchID]+=radius
                count_arr[branchID]+=1
                curve_rad_arr[i][j][k]=radius
    
    radius_dist_arr=np.divide(radius_dist_arr,count_arr)
    radius_dist_arr[0]=0
    return radius_dist_arr



prefix='img'

#loops over VOIs and compute vessel size distribution (VSD)

for i in range(1,2):
    ID=str(i)
    print('Processing Image: ',ID)

    '''
    Define all the name variables
    '''

    inputname = basepath+prefix+'_'+ID+'.hdr'
    grayscale_imgname = outpath+prefix+'_'+ID+'.nii.gz'
    binary_hole_imgname=basepath+prefix+'_'+ID+'_bin_hole.nii.gz'
    binary_imgname=outpath+prefix+'_'+ID+'_bin.nii.gz'
    eroded_imgname=outpath+prefix+'_'+ID+'_eroded_closed.nii.gz'
    cnnctd_imgname=outpath+prefix+'_'+ID+'_eroded_connected.nii.gz'
    skel_imgname=outpath+prefix+'_'+ID+'_skel_fiji.nrrd'
    radius_imgname=outpath+prefix+'_'+ID+'_rad3D.nii.gz'
    label_skel_imgname=outpath+prefix+'_'+ID+'_simple_branch.nii.gz'
    
    '''
    Reading the Raw image file
    '''

    grayimg = SimpleITK.ReadImage(inputname)
    grayarr = SimpleITK.GetArrayFromImage(grayimg)

    '''
    Elevating image contrast using CLAHE
    '''

    elevated_pix,vesselness = Vesselness_sensitive_CE_3D(grayarr,1,45,3,1) 
    Elev_CLAHE_img = SimpleITK.GetImageFromArray(elevated_pix)
    SimpleITK.WriteImage(Elev_CLAHE_img,grayscale_imgname)


    '''
    Applying Li's thresholding algorithm on the contrast elevated image 
    '''
    Elev_CLAHE_img = SimpleITK.ReadImage(grayscale_imgname)
    binarr = SimpleITK.GetArrayFromImage(Elev_CLAHE_img)
    binthresh = threshold_li(binarr) #change threshold_li to threshold_otsu if needed
    binarr[binarr<binthresh]=0
    binarr[binarr>0]=255
    print(binarr.shape)
    binimg0 = SimpleITK.GetImageFromArray(binarr)
    SimpleITK.WriteImage(binimg0, binary_hole_imgname)
    
    '''
    Morpholigical closing along with dilation to close the holes inside larger vessels
    '''
        
    bin_img = SimpleITK.ReadImage(binary_hole_imgname)
    binarr = SimpleITK.GetArrayFromImage(bin_img)
    closed_arr = ndimage.binary_closing(binarr,iterations=1).astype(int) #Dialation with a spere of radius 1 micron
    binimg = SimpleITK.GetImageFromArray(closed_arr)
    SimpleITK.WriteImage(binimg, binary_imgname)
    

    '''
    Morpholigical erosion on the binary segmentation using a kernel of radius 1  
    '''
    # 
    filter = SimpleITK.BinaryErodeImageFilter()
    filter.SetKernelRadius( 1 )
    filter.SetForegroundValue( 1 )
    eroded_image = filter.Execute ( binimg )
    SimpleITK.WriteImage(eroded_image, eroded_imgname)

    '''
    Extracting the maximally connected vascular structure from the holefilled binary image 
    '''
    eroded_image = SimpleITK.ReadImage(eroded_imgname)
    eroded_arr = SimpleITK.GetArrayFromImage(eroded_image)
    size = eroded_arr.shape

    labeled_image, count = skimage.measure.label(eroded_arr, return_num=True)
    objects = skimage.measure.regionprops(labeled_image)
    object_areas = [obj["area"] for obj in objects]
    max_area= np.max(object_areas)
    for obj in objects:
        if obj["area"]!=max_area: #Only keep the largest connected component
            coords = obj["coords"]
            for coord in coords:
                eroded_arr[coord[0]][coord[1]][coord[2]]=0
    #print('Here')

    cnctdimg = SimpleITK.GetImageFromArray(eroded_arr)
    SimpleITK.WriteImage(cnctdimg, cnnctd_imgname)
    
    '''
    Skeletonize the hole-filled binary image
    '''
    cnnctd_image = SimpleITK.ReadImage(cnnctd_imgname)
    cnnctd_arr = SimpleITK.GetArrayFromImage(cnnctd_image)
    skeleton_arr = skeletonize(cnnctd_arr)
    skeleton = SimpleITK.GetImageFromArray(skeleton_arr)
    SimpleITK.WriteImage(skeleton, skel_imgname)

    '''
    Labels each branch of the skeleton and then label the loops
    '''

    branchcnt=Extract_Skeleton_Info(skel_imgname,label_skel_imgname) 

    '''
    Compute the radius at each skeletal points
    '''
    command = 'build/RAD.exe '+skel_imgname+' '+cnnctd_imgname+' '+radius_imgname
    os.system(command)
    radius_image = SimpleITK.ReadImage(radius_imgname)
    radius_arr = SimpleITK.GetArrayFromImage(radius_image)
    size=radius_arr.shape


    '''
    Calculate the vascular volume fraction of the segmented structure 
    '''
    eroded_arr = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(cnnctd_imgname))
    non_zero_cnt = np.count_nonzero(eroded_arr)
    total_cnt = size[0]*size[1]*size[2]
    vf=round(100*(non_zero_cnt/total_cnt),1)
    print('Volume Fraction for '+ID+': '+str(vf))
    if vf<8:
        vfile.write('Volume Fraction for img'+ID+': '+str(vf)+'\n')
    print('Radius computation finished.....')

    '''
    Compute the vessel size distribution from the labelled skeletons and radius map 
    '''
    maxR_cutoff = 40
    base_fs_dist = np.zeros((maxR_cutoff+1))
    radius_arr = GetVesselSizeDistribution(label_skel_imgname,radius_imgname)
    for rad in radius_arr:
        intrad = int(np.round(rad))
        if intrad>maxR_cutoff:
            base_fs_dist[maxR_cutoff]+=1
        else:
            base_fs_dist[intrad]+=1
    
    print('Computed VSD: ',base_fs_dist)

