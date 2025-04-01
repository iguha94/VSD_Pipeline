import SimpleITK
import numpy as np 

def Compute_Width_3D(skeletonname,eroded_image_name,radius_image_name):

    skeleton = SimpleITK.ReadImage(skeletonname)
    skelarr = SimpleITK.GetArrayFromImage(skeleton)

    eroded = SimpleITK.ReadImage(eroded_image_name)
    erodedarr = SimpleITK.GetArrayFromImage(eroded)

    size = skelarr.shape
    radius_info = np.zeros(size)

    deg_step = 10
    phi_step = 10
    deg_tot = 360
    flag=1

    for z in range(1,size[0]-1):
        for x in range(size[1]):
            for y in range(size[2]):
                if skelarr[z][x][y]>0:
                    rad_min = 99999999
                    for phi in range(0,180,phi_step):
                        phi_rad = (3.14/180.0)*phi
                        for theta in range(0,360,deg_step):
                            theta_rad = (3.14/180.0)*theta
                            r=1
                            while(True):
                                z_off = int(np.ceil(z+(r*np.cos(phi_rad))))
                                x_off = int(np.ceil(x+(r*np.cos(theta_rad)*np.sin(phi_rad))))
                                y_off = int(np.ceil(y+(r*np.sin(phi_rad)*np.sin(theta_rad))))
                                if z_off>=0 and z_off<size[0] and x_off>=0 and x_off<size[1] and y_off>=0 and y_off<size[2]:
                                    if erodedarr[z_off][x_off][y_off]>0:
                                        r=r+1
                                    else:
                                        if rad_min>r:
                                            rad_min=r
                                        break
                                else:
                                    if rad_min>r:
                                        rad_min=r
                                    break
                    radius_info[z][x][y]=rad_min


    radius_img = SimpleITK.GetImageFromArray(radius_info)
    SimpleITK.WriteImage(radius_img,radius_image_name)

def Compute_Radius_3D(skeletonname,eroded_image_name,radius_image_name):
    print('Skeleton Radius Computation Started\n')
    skeleton = SimpleITK.ReadImage(skeletonname)
    skelarr = SimpleITK.GetArrayFromImage(skeleton)

    eroded = SimpleITK.ReadImage(eroded_image_name)
    erodedarr = SimpleITK.GetArrayFromImage(eroded)

    size = skelarr.shape
    radius_info = np.zeros(size)

    deg_step = 10
    phi_step = 10
    deg_tot = 360
    flag=1

    for z in range(1,size[0]-1):
        for x in range(size[1]):
            for y in range(size[2]):
                if skelarr[z][x][y]>0:
                    rad_min = 99999999
                    for phi in range(0,180,phi_step):
                        phi_rad = (3.14/180.0)*phi
                        for theta in range(0,360,deg_step):
                            theta_rad = (3.14/180.0)*theta
                            r=1
                            r1=0
                            while(True):
                                z_off = int(np.ceil(z+(r*np.cos(phi_rad))))
                                x_off = int(np.ceil(x+(r*np.cos(theta_rad)*np.sin(phi_rad))))
                                y_off = int(np.ceil(y+(r*np.sin(phi_rad)*np.sin(theta_rad))))
                                if z_off>=0 and z_off<size[0] and x_off>=0 and x_off<size[1] and y_off>=0 and y_off<size[2]:
                                    if erodedarr[z_off][x_off][y_off]>0:
                                        r=r+1
                                        if r>40:
                                            r1=r
                                            break
                                    else:
                                        r1=r
                                        break
                                else:
                                    r1=r
                                    break
                            
                            r=1
                            r2=0
                            while(True):
                                z_off = int(np.ceil(z-(r*np.cos(phi_rad))))
                                x_off = int(np.ceil(x-(r*np.cos(theta_rad)*np.sin(phi_rad))))
                                y_off = int(np.ceil(y-(r*np.sin(phi_rad)*np.sin(theta_rad))))
                                if z_off>=0 and z_off<size[0] and x_off>=0 and x_off<size[1] and y_off>=0 and y_off<size[2]:
                                    if erodedarr[z_off][x_off][y_off]>0:
                                        r=r+1
                                        if r>40:
                                            r2=r
                                            break
                                    else:
                                        r2=r
                                        break
                                else:
                                    r2=r
                                    break
                        radius = 0.5*(r1+r2)  
                        if rad_min>radius:
                            rad_min=radius            
                    radius_info[z][x][y]=rad_min


    radius_img = SimpleITK.GetImageFromArray(radius_info)
    SimpleITK.WriteImage(radius_img,radius_image_name)

    print('\nSkeleton Radius Computation Done ..... ')

