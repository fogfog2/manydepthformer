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

USE_FOV = False
USE_3d_VIEW = False
USE_Axis = True
USE_SPHERE = False


dir_base = "result/colon3/image_cmt"
intrinsic_dir = dir_base+"_intrinsic"
pose_dir = dir_base+"_pose"
rgb_dir = dir_base+"_rgb"

depth_list = sorted(os.listdir(dir_base))
intrinsic_list = sorted(os.listdir(intrinsic_dir))
pose_list = sorted(os.listdir(pose_dir))
rgb_list = sorted(os.listdir(rgb_dir))

pcd = o3d.geometry.PointCloud()
path_pcd = o3d.geometry.PointCloud()
view_pcd = o3d.geometry.PointCloud()
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
prev_pts = []


slam_map = []
slam_color = []
acc_pose = []

acc_colored = []
acc_point = []

def gen_occupied_color(occupied, center, mode=0):
    occupied_color =np.array(occupied,copy=True)
    
        
    dx = occupied[:,0] - center[0]
    dy = occupied[:,1]  - center[1]
    dz = occupied[:,2]  - center[2]
    
    dx2 = np.square(dx) 
    dist = np.sqrt( np.square(dx) + np.square(dy) + np.square(dz))
    
    if mode==0:
        value = cm.hot(dist/1.5)
    else:
        value = cm.jet(dist/1.5)
    
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

def path_loader_inv(pts):

    min_z = pts[:,2].min()
    max_z = pts[:,2].max()
    step = 10
    res = (max_z - min_z)/step

    path = [[0,0,0]]
    #path = []
    for i in range(9,step):
        step_in = max_z - res*(i+1) 
        step_out = max_z - res*(i)
        idx = np.where(pts[:,2]< step_out)
        subptx = pts[idx]
        idx = np.where(subptx[:,2]>=step_in)
        test = subptx[idx]
        x = test[:,0].mean()
        y = test[:,1].mean()
        z = (step_in + step_out)/2.0
        path.append([x,y,z])
    return path
def inference():
    global pcd
    global path_pcd
    global counter
    global fov_set
    global T
    global Local_T
    global slam_map, slam_color
    global octree_line_set
    global acc_pose
    global point_sphere
    global arrow_line_set
    global prev_pts
    temp_pcd = o3d.geometry.PointCloud()
    if len(depth_list)-1 < counter:                
        counter = 0
        Local_T=init_pose
        slam_map =[]
        slam_color = []
        acc_pose = []
        

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
    #colored = colored[0:-1:scale_x, 0:-1:scale_y, :]
    colored = colored.reshape(-1,3)
    #show pcd point
    #if SHOW_UNPROJECTED_DEPTH:  
    
    #fov
    if(USE_FOV):
        fov, _, fov_color = set_fov_line()
        fov_set.points=o3d.utility.Vector3dVector(fov)
        fov_set.colors=o3d.utility.Vector3dVector(np.float64(fov_color))
    
    #pred_depth
    
    cam_points, view_pts = backproject_depth(pred_depth, invk)    
    test = cam_points.cpu()[0].numpy()
    view = view_pts.cpu()[0].numpy()
    view = np.transpose(view).astype(np.float64)
    # scale = 16
    test  = test[0:3,:].reshape(3, WIDTH,HEIGHT)
    #test = test[:,0:-1:scale_x, 0:-1:scale_y]
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
    #mask = streamarray[:,2]<3.0
    #streamarray = streamarray[mask]
    #colored= colored[mask]
    is_key_frame = False
    
    #T, xyzs = update_pose(T, np_pose)
    Local_T = init_pose
    for pose in acc_pose:
        Local_T , _ = update_pose(Local_T, np_pose)
    
    is_key_frame =  keyframe_detector(Local_T)

    temp_pcd.points = o3d.utility.Vector3dVector(streamarray)
    temp_pcd.colors = o3d.utility.Vector3dVector(colored)    
        
    temp_pcd.transform(Local_T)
    
    if(USE_FOV):
        fov_set.transform(Local_T)

    current_map = np.array(temp_pcd.points)
    current_color = np.array(temp_pcd.colors)

    slam_map.append(current_map)
    slam_color.append(current_color)
    is_key_frame = True    
    acc_pose.append(np_pose)
    acc_range = 5
    if len(slam_map)>acc_range:
        test_map = slam_map[-acc_range:]
        test_pose = acc_pose[-acc_range:]
    else:
        test_map = slam_map
        test_pose = acc_pose
    
    resolution = 0.05
    octree = octomap.OcTree(resolution)
    if is_key_frame:            
        Local_T = init_pose
        
        for map, pose in zip(test_map,test_pose):    
            

            Local_T , _ = update_pose(Local_T, pose)
            octree.insertPointCloud(
            pointcloud=map,
            origin=np.array([Local_T[0][3], Local_T[1][3], Local_T[2][3]], dtype=float),
            maxrange=-1,
            )
            
        occupied, empty = octree.extractPointCloud()          
        occupied_color = np.array(occupied,copy=True)
        empty_color = np.array(empty,copy=True)
    
        aabb_min = octree.getMetricMin()
        aabb_max = octree.getMetricMax()
        
        octree_pt = np.array([ [aabb_min[0], aabb_min[1], aabb_min[2]], [aabb_min[0], aabb_max[1], aabb_min[2]] , [aabb_max[0], aabb_min[1], aabb_min[2]], [aabb_max[0], aabb_max[1], aabb_min[2]], [aabb_min[0], aabb_min[1], aabb_max[2]], [aabb_min[0], aabb_max[1], aabb_max[2]] , [aabb_max[0], aabb_min[1], aabb_max[2]], [aabb_max[0], aabb_max[1], aabb_max[2]]], dtype=np.float32)
        octree_line_set.points = o3d.utility.Vector3dVector(octree_pt)
        center = [Local_T[0][3], Local_T[1][3], Local_T[2][3]]
        occupied_color = gen_occupied_color(occupied, center)
        empty_color = gen_occupied_color(empty, center, 1)
        #occupied_color.fill(0.9)
        #empty_color.fill(0.7)
       
        dpose = empty - center
        dist =  dpose[:,0]*dpose[:,0]+dpose[:,1]*dpose[:,1]+dpose[:,2]*2*dpose[:,2]        
        sqrtdist = np.sqrt(dist)
        max_distanc_idx =sqrtdist.argmax()
        max_point= empty[max_distanc_idx]
        
        occupied_np = np.concatenate( [occupied, empty])
        occupied_color_np = np.concatenate( [occupied_color, empty_color])
        
        # pcd.points = o3d.utility.Vector3dVector(occupied)
        # pcd.colors = o3d.utility.Vector3dVector(occupied_color)015
            
        # path_pcd.points = o3d.utility.Vector3dVector(empty)
        # path_pcd.colors = o3d.utility.Vector3dVector(empty_color)
        
        # pcd.points = o3d.utility.Vector3dVector(empty)
        # pcd.colors = o3d.utility.Vector3dVector(empty_color)
        
        #vis.update_geometry(pcd)   
        
        render = vis.get_render_option()
        render.point_size = 5.0    

    
    max_point = max_point * 3
    
    if len(prev_pts)==0:
        current_pts = [max_point[0],max_point[1],max_point[2]/2.0]
    else:
        current_pts = [ (max_point[0] + prev_pts[0])/2.0, (max_point[1]+prev_pts[1])/2.0, (max_point[2]/2.0 + prev_pts[2])/2.0]
    
    #occupancy path
    if(USE_SPHERE):
        point_sphere.translate( current_pts, relative=False )
    arrow_pts = np.array([current_pts ,  [center[0],center[1],center[2] ]  ]  )
    arrow_line_set.points = o3d.utility.Vector3dVector(arrow_pts)   
    prev_pts= current_pts 
    
    #depth path
    
    
    
    #if counter ==0:
    #slam_map.append(current_map)
    #slam_color.append(current_color)
    slam_map = slam_map[-acc_range:]
    acc_pose = acc_pose[-acc_range:]
    slam_map_np = np.concatenate(slam_map)
    slam_color_np = np.concatenate(slam_color)
    
    pts = np.asarray(current_map)    
    arrow2 = path_loader_inv(pts)    
    
    global arrow2_line_set
    
    if(len(arrow2)>0):
        for i in range(len(arrow2)):
            arrow2[i][0] = arrow2[i][0] *3
            arrow2[i][1] = arrow2[i][1] *3
            arrow2[i][2] = arrow2[i][2] *3
            
    arrow2_line_set.points = o3d.utility.Vector3dVector(arrow2)
    
    # pcd.points = o3d.utility.Vector3dVector(slam_map_np)
    # pcd.colors = o3d.utility.Vector3dVector(slam_color_np)   
    
    if(USE_3d_VIEW):
        pcd.points = o3d.utility.Vector3dVector(current_map)
        pcd.colors = o3d.utility.Vector3dVector(current_color)   

    view[:,2] = -0.5
    view_pcd.points = o3d.utility.Vector3dVector(view)
    view_pcd.colors = o3d.utility.Vector3dVector(current_color)   
    
    print(T[0][3], T[1][3], T[2][3])
    
    if(USE_SPHERE):
        vis.update_geometry(point_sphere)
    if(USE_3d_VIEW):
        vis.update_geometry(pcd)   
    vis.update_geometry(octree_line_set)
    vis.update_geometry(path_pcd)   
    if(USE_FOV):
        vis.update_geometry(fov_set)
    vis.update_geometry(arrow_line_set)
    vis.update_geometry(arrow2_line_set)
    vis.update_geometry(view_pcd)
    vis.update_renderer()        
    vis.poll_events()
    counter = counter+1


def animation_callback(vis):  
    global counter
    if len(depth_list)-1 > counter:         
        
        inference()
        #counter = len(depth_list)-1


#FOV area
if(USE_FOV):
    fov, fov_lines, fov_color = set_fov_line()
    fov_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(fov),
        lines=o3d.utility.Vector2iVector(fov_lines)
    )    
    fov_set.colors=o3d.utility.Vector3dVector(np.float64(fov_color))
    fov_set.transform(T)

#axis 
if(USE_Axis):
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

#point sphere
point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=20)
point_sphere.paint_uniform_color([1,0.0,0])

#poin line
arrow_pt = np.array([ [0, 0, 0], [0,0,0] ], dtype=np.float32)
arrow_lines = [[0,1]] 
arrow_color= [[0,0,1]]
arrow_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(arrow_pt),
    lines=o3d.utility.Vector2iVector(arrow_lines)
    
)
arrow_line_set.colors=o3d.utility.Vector3dVector(arrow_color)

#axis line
arrow2_pt = np.array([ [0, 0, 0], [0,0,0] ], dtype=np.float32)
arrow2_lines = [[0,1]] 
arrow2_color= [[1,0,0]]
arrow2_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(arrow2_pt),
    lines=o3d.utility.Vector2iVector(arrow2_lines)
    
)
arrow2_line_set.colors=o3d.utility.Vector3dVector(arrow2_color)



if __name__ == "__main__":

    vis.create_window()    
    vis.add_geometry(path_pcd)
    vis.add_geometry(pcd)    
    if(USE_Axis):
        vis.add_geometry(axis_line_set)    
    if(USE_FOV):
        vis.add_geometry(fov_set)
    if(USE_SPHERE):
        vis.add_geometry(point_sphere)
    vis.add_geometry(arrow_line_set)
    vis.add_geometry(arrow2_line_set)
    vis.add_geometry(view_pcd)
    #vis.add_geometry(octree_line_set)
    vis.register_animation_callback(animation_callback)
    #vis.register_key_callback(65,animation_callback)
    vis.run()

