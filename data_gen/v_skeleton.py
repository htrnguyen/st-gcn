import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import sys
import cv2

sys.path.extend(['../'])
from .ntu_gendata import read_xyz
from .preprocess import pre_normalization


inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
edge_info = [(i - 1, j - 1) for (i, j) in inward_ori_index]



def plot_skeleton(vertex_info, edge_info):
    ax = plt.axes(projection="3d")
    for i in range(0,81,81):
        x,y,z = vertex_info[i][:,0],vertex_info[i][:,1],vertex_info[i][:,2]
        ax.scatter(x,y,z, c='r',s=100)
        for edge in edge_info:
            x_p = [x[edge[0]],x[edge[1]]]
            y_p = [y[edge[0]],y[edge[1]]]
            z_p = [z[edge[0]],z[edge[1]]]
            if i == 0: 
                c_e = 'b'
            elif i == 27:
                c_e = 'y'
            else:
                c_e = 'g'
            ax.plot(x_p,y_p,z_p, color=c_e)
    plt.show()

def draw_skeleton(name,skeletons,edge=edge_info,label=''):
    imgs = []
    center = (270,270)
    for skeleton in skeletons:
        img = np.zeros((540,540,3))
        x_root=540-int((skeleton[1][0]+0.5)*540/2)
        y_root=540-int((skeleton[1][1]+0.5)*540/2)
        root = (center[0] - x_root,center[1] - y_root)
        for i,j in edge:
            xi=540-int((skeleton[i][0]+0.5)*540/2)
            yi=540-int((skeleton[i][1]+0.5)*540/2)
            xj=540-int((skeleton[j][0]+0.5)*540/2)
            yj=540-int((skeleton[j][1]+0.5)*540/2)
            cv2.line(img,(xi+root[0],yi+root[1]),(xj+root[0],yj+root[1]),(255,255,255),2)
        if label != '':
            if label != '':
                img = cv2.putText(img, label, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,215,0), 3, cv2.LINE_AA)
        imgs.append(img)
        
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(name, fourcc,30,(540,540))
    for i in range(len(imgs)):
        out.write(imgs[i].astype('uint8'))
    out.release()

# info = read_xyz('D:\Hoc-tap\KLTN\source-code\GCN-for-Human-Action-Recognition\data\S001C001P001R001A001.skeleton')
# C, T, V, M = info.shape






# # visualize
# # skeleton orignal
# vertex_info = np.transpose(info, [3, 1, 2, 0])[0]
# plot_skeleton(vertex_info,edge_info)

# # normalization skeleton 
# fp = np.zeros((1, 3,300, 25, 2), dtype=np.float32)
# fp[:,:,0:info.shape[1],:,:]=info.reshape(1,C,T,V,M)
# vertex_info_v = np.transpose(pre_normalization(fp)[0], [3, 1, 2, 0])[0]
# plot_skeleton(vertex_info_v,edge_info)

# print(pre_normalization(info.reshape(1,C,T,V,M)).shape)