import os
import numpy as np
import open3d as o3d
import time
import cv2
from manydepth.layers import disp_to_depth, BackprojectDepth
import torch

dir_base = "image_cmt"
intrinsic_dir = dir_base+"_intrinsic"
pose_dir = dir_base+"_pose"
rgb_dir = dir_base+"_rgb"

depth_list = sorted(os.listdir(dir_base))
intrinsic_list = sorted(os.listdir(intrinsic_dir))
pose_list = sorted(os.listdir(pose_dir))
rgb_list = sorted(os.listdir(rgb_dir))

pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.VisualizerWithKeyCallback()
counter = 0
WIDTH  = 640
HEIGHT = 192
backproject_depth = BackprojectDepth(1, HEIGHT, WIDTH)


SHOW_UNPROJECTED_DEPTH = True
init_pose = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
T = init_pose

def update_pose(PrevT, T):
    cam_to_world = np.dot( PrevT , T)
    xyzs = cam_to_world[:3, 3]

    return cam_to_world, xyzs
def load_image_my(image_path, depth_path):
    color_raw = o3d.io.read_image(image_path)
    pp =  np.concatenate((np.array(color_raw)[:,:,0].reshape(256,256,1), np.array(color_raw)[:,:,1].reshape(256,256,1), np.array(color_raw)[:,:,2].reshape(256,256,1)),axis=2)
    color_raw = o3d.geometry.Image(pp)
    depth_raw = o3d.io.read_image(depth_path)
    return color_raw, depth_raw

def loader(counter):

    d_image =cv2.imread(dir_base+"/"+depth_list[counter],0)
    d_image = (d_image/256.0)*11.0

    rgb_image =cv2.imread(rgb_dir+"/"+rgb_list[counter])
    intrinsic_file = open(intrinsic_dir+"/"+intrinsic_list[counter],'r')
    pose_file = open(pose_dir+"/"+pose_list[counter],'r')
    intrinsic_array = intrinsic_file.readline().split(",")
    intrinsic_array=intrinsic_array[0:16]
    np_intrinsic_array = [float(i) for i in intrinsic_array]

    intrinsic_k_array = intrinsic_file.readline().split(",")
    intrinsic_k_array=intrinsic_k_array[0:16]
    np_intrinsic_k_array = [float(i) for i in intrinsic_k_array]

    pose_array = pose_file.readline().split(",")

    pose_array=pose_array[0:16]
    np_pose_array = [float(i) for i in pose_array]
    return d_image, rgb_image, np_intrinsic_array,np_intrinsic_k_array, np_pose_array

def set_fov_line():
    fov_center = [0,0,0]
    near = 0.12
    far = 0.5
    width = 0.43
    height = 0.13
    ratio = near/far

    fov_near_lt = [-width*ratio, height*ratio, near]
    fov_near_lb = [-width*ratio, -height*ratio, near]
    fov_near_rt = [width*ratio, height*ratio, near]
    fov_near_rb = [width*ratio, -height*ratio, near]

    fov_far_lt = [-width, height, far]
    fov_far_lb = [-width, -height, far]
    fov_far_rt = [width, height, far]
    fov_far_rb = [width, -height, far]

    fov = [ fov_near_lt, fov_near_lb,fov_near_rb , fov_near_rt, fov_far_lt,fov_far_lb,fov_far_rb,fov_far_rt, fov_center]
    fov_lines = [[0,1], [1,2],[2,3], [0,3], [4,5],[5,6],[6,7],[4,7],[4,8],[5,8],[6,8],[7,8]] 
    fov_color = [[0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0]]
    return fov, fov_lines, fov_color

def inference():
    #global pcd
    global counter
    global fov_set
    global T


    pcd = o3d.geometry.PointCloud()
    if len(depth_list)-1 < counter:                
        counter = 0

    d_image, rgb_image, intrinsic_array, intrinsic_k_array, pose_array = loader(counter)



    #depth 
    np_pred_depth = cv2.resize(d_image, (WIDTH, HEIGHT))
    pred_depth = torch.Tensor(np_pred_depth)
    pred_depth = torch.unsqueeze(pred_depth, 0)
    pred_depth = torch.unsqueeze(pred_depth, 0)
    
    #inv_k
    np_intrinsic = np.array(intrinsic_array)
    invk = torch.Tensor(np_intrinsic)
    invk = torch.reshape(invk, (1,4,4))

    #k
    np_k_intrinsic = np.array(intrinsic_k_array)
    #k = torch.Tensor(np_k_intrinsic)
    #k = torch.reshape(k, (1,4,4))

    #pose
    np_pose = np.array(pose_array)
    np_pose = np.reshape(np_pose,(4,4))
    #pose = torch.Tensor(np_pose)
    #pose = torch.reshape(pose, (1,4,4))

    #rgb
    rgb_image = cv2.resize(rgb_image, (WIDTH, HEIGHT))
    colored = rgb_image.reshape(rgb_image.shape[0]*rgb_image.shape[1],3)
    colored = np.float64(colored)
    colored = colored[...,::-1]/255.0

    #show pcd point
    #if SHOW_UNPROJECTED_DEPTH:  
    
    #fov
    fov, _, fov_color = set_fov_line()
    fov_set.points=o3d.utility.Vector3dVector(fov)
    fov_set.colors=o3d.utility.Vector3dVector(np.float64(fov_color))

    cam_points = backproject_depth(pred_depth, invk)    
    test = cam_points.cpu()[0].numpy()
    

    value_x = test[0,:]
    value_y = test[1,:]
    value_z = test[2,:]
    streamarray=[]
    streamarray.append(value_x)
    streamarray.append(value_y)
    streamarray.append(value_z)
    streamarray = np.concatenate(streamarray)
    streamarray = np.reshape(streamarray, (3,-1))
    streamarray = np.transpose(streamarray)        
    
    mask = streamarray[:,2]<0.5
    streamarray = streamarray[mask]
    colored=colored[mask]
    
    # mask = streamarray[:,0]<0.0
    # streamarray = streamarray[mask]
    # colored=colored[mask]
    

    pcd.colors = o3d.utility.Vector3dVector(colored)
    pcd.points = o3d.utility.Vector3dVector(streamarray)
    T, xyzs = update_pose(T, np_pose)
    pcd.transform(T)
    fov_set.transform(T)
    vis.add_geometry(pcd)   
    vis.update_geometry(fov_set)
    counter = counter+1


def animation_callback(vis):  
    global counter
    if len(depth_list)-1 > counter:         
        inference()


#FOV area
fov, fov_lines, fov_color = set_fov_line()
fov_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(fov),
    lines=o3d.utility.Vector2iVector(fov_lines)
)    
fov_set.colors=o3d.utility.Vector3dVector(np.float64(fov_color))
fov_set.transform(T)

#axis 
axis_pt = np.array([ [0, 0, 0], [0,2,0] , [2,0,0], [0,0,2]], dtype=np.float32)

axis_lines = [[0,1],[0,2],[0,3]] 

axis_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(axis_pt),
    lines=o3d.utility.Vector2iVector(axis_lines),
)
axis_line_set.transform(T)


if __name__ == "__main__":

    vis.create_window()    
    vis.add_geometry(pcd)    
    vis.add_geometry(axis_line_set)    
    vis.add_geometry(fov_set)
    vis.register_animation_callback(animation_callback)
    vis.run()

