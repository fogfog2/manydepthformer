import os
import numpy as np
import open3d as o3d
import time
import cv2
from manydepth.layers import disp_to_depth, BackprojectDepth
import torch
import math
import octomap
import matplotlib.cm as cm

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
WIDTH  = 256
HEIGHT = 256
backproject_depth = BackprojectDepth(1, HEIGHT, WIDTH)


SHOW_UNPROJECTED_DEPTH = True
init_pose = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
T = init_pose
Local_T = init_pose
#octomap

resolution = 0.1
octree = octomap.OcTree(resolution)

slam_map = []
slam_color = []
def gen_occupied_color(occupied, center, mode=0):
    occupied_color =np.array(occupied,copy=True)
    
        
    dx = occupied[:,0] - center[0]
    dy = occupied[:,1]  - center[1]
    dz = occupied[:,2]  - center[2]
    
    dx2 = np.square(dx) 
    dist = np.sqrt( np.square(dx) + np.square(dy) + np.square(dz))
    
    if mode==0:
        value = cm.hot(dist/3.0)
    else:
        value = cm.jet(dist/3.0)
    
    occupied_color[:,0] = value[:,0]
    occupied_color[:,1] = value[:,1]
    occupied_color[:,2] = value[:,2]
    
        
    
    
    return occupied_color

def keyframe_detector(T):
    dist = math.sqrt(T[0][3]*T[0][3]+T[1][3]*T[1][3]+T[2][3]*T[2][3])
    
    if dist <0.01:
        return False
    else:
        return True

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
    # width = 0.43
    # height = 0.13
    width = 0.8
    height = 0.8
    
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
    global pcd
    global counter
    global fov_set
    global T
    global Local_T
    global slam_map, slam_color
    global octree_line_set
    
    #pcd = o3d.geometry.PointCloud()
    if len(depth_list)-1 < counter:                
        counter = 0
        Local_T=init_pose
        slam_map =[]
        slam_color = []
        

    d_image, rgb_image, intrinsic_array, intrinsic_k_array, pose_array = loader(counter)



    #depth 
    np_pred_depth = cv2.resize(d_image, (WIDTH, HEIGHT))
    kernel = np.ones((5,5), np.float32) / 25
    np_pred_depth = cv2.filter2D(np_pred_depth ,-1, kernel)
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
    
    
    scale_x = 1
    scale_y = 1
    
    colored = colored.reshape(WIDTH,HEIGHT,3)
    colored = colored[0:-1:scale_x, 0:-1:scale_y, :]
    colored = colored.reshape(-1,3)
    #show pcd point
    #if SHOW_UNPROJECTED_DEPTH:  
    
    #fov
    fov, _, fov_color = set_fov_line()
    fov_set.points=o3d.utility.Vector3dVector(fov)
    fov_set.colors=o3d.utility.Vector3dVector(np.float64(fov_color))
    
    #pred_depth
    
    cam_points = backproject_depth(pred_depth, invk)    
    test = cam_points.cpu()[0].numpy()
    
    # scale = 16
    test  = test[0:3,:].reshape(3, WIDTH,HEIGHT)
    test = test[:,0:-1:scale_x, 0:-1:scale_y]
    test = test.reshape(3,-1)
    value_x = test[0,:]
    value_y = test[1,:]
    value_z = test[2,:]
     
    streamarray=[]
    streamarray.append(value_x)
    streamarray.append(value_y)
    streamarray.append(value_z)
    
        #

    streamarray = np.concatenate(streamarray)
    streamarray = np.reshape(streamarray, (3,-1))
    streamarray = np.transpose(streamarray)
    streamarray = streamarray.astype(np.float64)
    mask = streamarray[:,2]<3.0
    streamarray = streamarray[mask]
    colored= colored[mask]
    is_key_frame = False
    
    T, xyzs = update_pose(T, np_pose)
    Local_T , _ = update_pose(Local_T, np_pose)
    
    
    is_key_frame =  keyframe_detector(Local_T)

    

    pcd.points = o3d.utility.Vector3dVector(streamarray)
    pcd.colors = o3d.utility.Vector3dVector(colored)    
        
    pcd.transform(T)
    fov_set.transform(T)

    current_map = np.array(pcd.points)
    current_color = np.array(pcd.colors)
    is_key_frame = True
    if is_key_frame:            
        slam_map.append(current_map)
        slam_color.append(current_color)
        Local_T = init_pose
        
        
        octree.insertPointCloud(
        pointcloud=current_map,
        origin=np.array([T[0][2], T[1][2], T[2][2]], dtype=float),
        maxrange=-1,
        )
        occupied, empty = octree.extractPointCloud()          
        occupied_color = np.array(occupied,copy=True)
        empty_color = np.array(empty,copy=True)
    
        aabb_min = octree.getMetricMin()
        aabb_max = octree.getMetricMax()
        
        octree_pt = np.array([ [aabb_min[0], aabb_min[1], aabb_min[2]], [aabb_min[0], aabb_max[1], aabb_min[2]] , [aabb_max[0], aabb_min[1], aabb_min[2]], [aabb_max[0], aabb_max[1], aabb_min[2]], [aabb_min[0], aabb_min[1], aabb_max[2]], [aabb_min[0], aabb_max[1], aabb_max[2]] , [aabb_max[0], aabb_min[1], aabb_max[2]], [aabb_max[0], aabb_max[1], aabb_max[2]]], dtype=np.float32)
        octree_line_set.points = o3d.utility.Vector3dVector(octree_pt)
        center = [T[0][2], T[1][2], T[2][2]]
        occupied_color = gen_occupied_color(occupied, center)
        empty_color = gen_occupied_color(empty, center, 1)
        #occupied_color.fill(0.9)
        #empty_color.fill(0.7)
       
       
         
        occupied_np = np.concatenate( [occupied, empty])
        occupied_color_np = np.concatenate( [occupied_color, empty_color])
        
        # pcd.points = o3d.utility.Vector3dVector(occupied)
        # pcd.colors = o3d.utility.Vector3dVector(occupied_color)015
            
        pcd.points = o3d.utility.Vector3dVector(occupied_np)
        pcd.colors = o3d.utility.Vector3dVector(occupied_color_np)
        
        # pcd.points = o3d.utility.Vector3dVector(empty)
        # pcd.colors = o3d.utility.Vector3dVector(empty_color)
        
        #vis.update_geometry(pcd)   
        
        render = vis.get_render_option()
        render.point_size = 50.0    
        
        
    if counter ==0:
        slam_map.append(current_map)
        slam_color.append(current_color)
    
    slam_map_np = np.concatenate(slam_map)
    slam_color_np = np.concatenate(slam_color)
    
    # pcd.points = o3d.utility.Vector3dVector(current_map)
    # pcd.colors = o3d.utility.Vector3dVector(current_color)   
    
    print(T[0][3], T[1][3], T[2][3])
    vis.update_geometry(octree_line_set)
    vis.update_geometry(pcd)   
    vis.update_geometry(fov_set)
    vis.update_renderer()        
    vis.poll_events()
    counter = counter+1


def animation_callback(vis):  
    global counter
    if len(depth_list)-1 > counter:         
        
        inference()
        #counter = len(depth_list)-1


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

#octree-box
octree_pt = np.array([ [0, 0, 0], [0,1,0] , [1,0,0], [1,1,0], [0, 0, 1], [0,1,1] , [1,0,1], [1,1,1]], dtype=np.float32)
octree_lines = [[0,1],[0,2],[0,4],[1,3],[1,5], [2,6],[3,7],[4,6],[5,7],[6,7]] 
octree_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(octree_pt),
    lines=o3d.utility.Vector2iVector(octree_lines),
)
octree_line_set.transform(T)

if __name__ == "__main__":

    vis.create_window()    
    vis.add_geometry(pcd)    
    vis.add_geometry(axis_line_set)    
    vis.add_geometry(fov_set)
    vis.add_geometry(octree_line_set)
    #vis.register_animation_callback(animation_callback)
    vis.register_key_callback(65,animation_callback)
    vis.run()

