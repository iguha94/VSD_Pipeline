import SimpleITK
import numpy as np
import skimage.measure

ISOLATED_POINT=0
END_POINT=1
SIMPLE_POINT=2
JUNCTION_POINT=3

def point_type(img,x,y,z,size):
    n=0
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if i==0 and j==0 and k==0:
                    continue
                #print(x+i,y+j,z+k,'--->',img[x+i][y+j][z+k])
                if x+i>=0 and x+i<size[0] and y+j>=0 and y+j<size[1] and z+k>=0 and z+k<size[2]:
                    if img[x+i][y+j][z+k]>0:
                        n=n+1
    if n==1:
        return END_POINT
    if n==2:
        return SIMPLE_POINT
    if n>=3:
        return JUNCTION_POINT
    return ISOLATED_POINT

def Extract_Skeleton_Info(skelname,simple_branch_name):

    skelimg = SimpleITK.ReadImage(skelname)
    skelarr = SimpleITK.GetArrayFromImage(skelimg)

    newskel=SimpleITK.GetImageFromArray(skelarr)
    SimpleITK.WriteImage(newskel,skelname)
    newskelarr = SimpleITK.GetArrayFromImage(newskel)

    size = skelarr.shape
    visited = np.zeros(size)
    brnachdict={}
    branchID=0
    flag=True
    while flag:
        flag=False 
        stack = []
        print('Reiterating...')
        epoint=0
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    # if not(x==158 and y==111 and z==116):
                    #     continue
                    if visited[x][y][z]==0 and skelarr[x][y][z]>0 and point_type(skelarr,x,y,z,size)==END_POINT:
                        epoint=epoint+1
                        flag=True
                        #visited[x][y][z]=1
                        stack.append((x,y,z))
                        branchlist=[]
                        branchID+=1
                        #print('Processing Branch: ',branchID,' ending at: ',x,y,z)
                        while len(stack)>0:
                            center=stack.pop()
                            xc=center[0]
                            yc=center[1]
                            zc=center[2]
                            visited[xc][yc][zc]=1
                            #print('Branch Length: ',len(branchlist))
                            #print('Processing point: ',xc,yc,zc,' Branch length: ',len(branchlist))
                            for i in range(-1,2):
                                for j in range(-1,2):
                                    for k in range(-1,2):
                                        if i==0 and j==0 and k==0:
                                            continue
                                        if xc+i>=0 and xc+i<size[0] and yc+j>=0 and yc+j<size[1] and zc+k>=0 and zc+k<size[2] and skelarr[xc+i][yc+j][zc+k]>0 and visited[xc+i][yc+j][zc+k]==0:
                                            
                                            type=point_type(skelarr,xc+i,yc+j,zc+k,size)
                                            #print('Next Point: ',xc+i,yc+j,zc+k,' with type: ',type)
                                            if type==SIMPLE_POINT:
                                                #visited[xc+i][yc+j][zc+k]=1
                                                #print('Point type: Simple')
                                                stack.append((xc+i,yc+j,zc+k))
                                                branchlist.append((xc+i,yc+j,zc+k))
                                            elif type==END_POINT:
                                                #print('Point type: End')
                                                visited[xc+i][yc+j][zc+k]=1
                                                branchlist.append((xc+i,yc+j,zc+k))
                                            elif type==JUNCTION_POINT:
                                                # print('Point type: Junction')
                                                # print('Ending at Junction Point: ',xc+i,yc+j,zc+k)
                                                branchlist.append((xc+i,yc+j,zc+k))

                        brnachdict[branchID]=branchlist
                        for tuple in branchlist:
                            skelarr[tuple[0]][tuple[1]][tuple[2]]=0

    print('All the Branches have been Extracted......')
    print('Detecting Loops......')

    looparr = np.zeros(size) 
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                looparr[x][y][z]=skelarr[x][y][z]

    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                if looparr[x][y][z]>0:
                    branchlist=[]
                    if point_type(looparr,x,y,z,size)==ISOLATED_POINT:
                        looparr[x][y][z]=0

   
    simplebranch_arr = np.zeros(size) 
    for key in brnachdict:
        color=key
        list_tuples=brnachdict[key]
        for tuple in list_tuples:
            simplebranch_arr[tuple[0]][tuple[1]][tuple[2]]=color
    
    labeled_skel, count = skimage.measure.label(looparr, return_num=True)
    objects = skimage.measure.regionprops(labeled_skel)
    for obj in objects:
        coords = obj["coords"]
        branchlist=[]
        for coord in coords:
            branchlist.append(coord)
            simplebranch_arr[coord[0]][coord[1]][coord[2]]=branchID
        branchID+=1
        brnachdict[branchID]=branchlist
    
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                if looparr[x][y][z]>0:
                    skelarr[x][y][z]=0
    
    stack = []
    visited = np.zeros(size)
    for x in range(size[0]):
        for y in range(size[1]):
            for z in range(size[2]):
                if simplebranch_arr[x][y][z]>0:
                    stack.append((x,y,z))
                    visited[x][y][z]=1

    while len(stack)>0:
        center=stack.pop()
        xc=center[0]
        yc=center[1]
        zc=center[2]
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if i==0 and j==0 and k==0:
                        continue
                    if xc+i>=0 and xc+i<size[0] and yc+j>=0 and yc+j<size[1] and zc+k>=0 and zc+k<size[2] and skelarr[xc+i][yc+j][zc+k]>0:
                        simplebranch_arr[xc+i][yc+j][zc+k]=simplebranch_arr[xc][yc][zc]
                        stack.append((xc+i,yc+j,zc+k))
                        skelarr[xc+i][yc+j][zc+k]=0
                                            
        

    simplebranch_img=SimpleITK.GetImageFromArray(simplebranch_arr)
    # skel_loop_img = SimpleITK.GetImageFromArray(skelarr)

    SimpleITK.WriteImage(simplebranch_img,simple_branch_name)
    # SimpleITK.WriteImage(skel_loop_img,loop_name)
    print('Finished Detecting Loops ......')
    return branchID
