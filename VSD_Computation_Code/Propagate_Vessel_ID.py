import SimpleITK
import numpy as np

def cartesianDistance(a,b,c):
    return int(np.sqrt(a*a+b*b+c*c)*10.0)

def mark_Vessels(labelled_skel_arr,eroded_arr,radius_arr,label_vessel_imgname,vessel_radius_imgname):
    print('Propagating Vessel Radius.......')
    size = labelled_skel_arr.shape
    labelled_vessel_arr = np.zeros(size)
    vessel_radius_arr = np.zeros(size)
    dist_arr = np.zeros(size)+9999999
    print(np.min(dist_arr))
    stack=[]

    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if labelled_skel_arr[i][j][k]>0:
                    labelled_vessel_arr[i][j][k]=labelled_skel_arr[i][j][k]
                    vessel_radius_arr[i][j][k] = radius_arr[i][j][k]
                    dist_arr[i][j][k]=0
                    stack.append((i,j,k,labelled_skel_arr[i][j][k]))
    
    flag=True
    while len(stack)>0:
        #print('Stack Size: ',len(stack))
        center=stack.pop(0)
        xc=center[0]
        yc=center[1]
        zc=center[2]

        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if i==0 and j==0 and k==0:
                        continue
                    if xc+i>=0 and xc+i<size[0] and yc+j>=0 and yc+j<size[1] and zc+k>=0 and zc+k<size[2] and eroded_arr[xc+i][yc+j][zc+k]>0 and \
                        labelled_skel_arr[xc+i][yc+j][zc+k]==0 and dist_arr[xc+i][yc+j][zc+k]>dist_arr[xc][yc][zc]+cartesianDistance(i,j,k):
                        dist_arr[xc+i][yc+j][zc+k]=dist_arr[xc][yc][zc]+cartesianDistance(i,j,k)
                        labelled_vessel_arr[xc+i][yc+j][zc+k] = labelled_vessel_arr[xc][yc][zc]
                        vessel_radius_arr[xc+i][yc+j][zc+k] = vessel_radius_arr[xc][yc][zc]
                        stack.append((xc+i,yc+j,zc+k,labelled_vessel_arr[xc+i][yc+j][zc+k]))


    Vessel_img=SimpleITK.GetImageFromArray(labelled_vessel_arr)
    SimpleITK.WriteImage(Vessel_img,label_vessel_imgname)

    Vessel_rad_img=SimpleITK.GetImageFromArray(vessel_radius_arr)
    SimpleITK.WriteImage(Vessel_rad_img,vessel_radius_imgname)
