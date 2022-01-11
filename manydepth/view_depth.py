# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from manydepth.utils import readlines
from manydepth.options import MonodepthOptions
from manydepth import datasets, networks
from manydepth.layers import transformation_from_parameters, disp_to_depth, BackprojectDepth
import tqdm

import open3d as o3d

import time
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

g_height = 0
g_width = 0
pred_disps = []
inv_ks = []
stream=[]

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set

from manydepth.utils import readlines
from manydepth.options import MonodepthOptions
from manydepth import datasets, networks
from manydepth.layers import transformation_from_parameters, disp_to_depth
import tqdm
    """
    global pred_disps
    global inv_ks
    global g_height
    global g_width
    global stream
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    device = torch.device("cuda")
    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # Setup dataloaders
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        filenames = filenames[::2]
        if opt.eval_teacher:
            encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
            encoder_class = networks.ResnetEncoder

        else:
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

            #encoder_model = "resnet" 
            #encoder_model = "swin_h" 
            encoder_model = "cmt_h"
            
            if "resnet" in encoder_model:            
                encoder_class = networks.ResnetEncoderMatching
            elif "swin_h" in encoder_model:
                encoder_class = networks.SwinEncoderMatching
            elif "cmt_h" in encoder_model:
                encoder_class = networks.CMTEncoderMatching
                    
            #encoder_class = networks.ResnetEncoderMatching

        encoder_dict = torch.load(encoder_path)
        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width
        g_height = HEIGHT
        g_width = WIDTH

        img_ext = '.png' if opt.png else '.jpg'
        if opt.eval_split == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False,
                                                     img_ext=img_ext)

        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               frames_to_load, 4,
                                               is_train=False,
                                               img_ext=img_ext)

        
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        # setup models
        if opt.eval_teacher:
            encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False)
        else:
            encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins)
            pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
            pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))

            pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
            pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                            num_frames_to_predict_for=2)

            pose_enc.load_state_dict(pose_enc_dict, strict=True)
            pose_dec.load_state_dict(pose_dec_dict, strict=True)

            min_depth_bin = encoder_dict.get('min_depth_bin')
            max_depth_bin = encoder_dict.get('max_depth_bin')

            pose_enc.eval()
            pose_dec.eval()

            if torch.cuda.is_available():
                pose_enc.cuda()
                pose_dec.cuda()

        encoder = encoder_class(**encoder_opts)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.eval()
        depth_decoder.eval()

        if torch.cuda.is_available():
            encoder.cuda()
            depth_decoder.cuda()

        backproject_depth = BackprojectDepth(1, g_height, g_width)
        backproject_depth.to("cuda")
        

        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color', 0, 0)]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()

                if opt.static_camera:
                    for f_i in frames_to_load:
                        data["color", f_i, 0] = data[('color', 0, 0)]

                # predict poses
                pose_feats = {f_i: data["color", f_i, 0] for f_i in frames_to_load}
                if torch.cuda.is_available():
                    pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
                # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                for fi in frames_to_load[1:]:
                    if fi < 0:
                        pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                        pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                        axisangle, translation = pose_dec(pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=True)

                        # now find 0->fi pose
                        if fi != -1:
                            pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                    else:
                        pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                        pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                        axisangle, translation = pose_dec(pose_inputs)
                        pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)

                        # now find 0->fi pose
                        if fi != 1:
                            pose = torch.matmul(pose, data[('relative_pose', fi - 1)])

                    data[('relative_pose', fi)] = pose

                lookup_frames = [data[('color', idx, 0)] for idx in frames_to_load[1:]]
                lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

                relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
                relative_poses = torch.stack(relative_poses, 1)

                K = data[('K', 2)]  # quarter resolution for matching
                invK = data[('inv_K', 2)]

                if torch.cuda.is_available():
                    lookup_frames = lookup_frames.cuda()
                    relative_poses = relative_poses.cuda()
                    K = K.cuda()
                    invK = invK.cuda()

                if opt.zero_cost_volume:
                    relative_poses *= 0

                if opt.post_process:
                    raise NotImplementedError

                output, lowest_cost, costvol = encoder(input_color, lookup_frames,
                                                        relative_poses,
                                                        K,
                                                        invK,
                                                        min_depth_bin, max_depth_bin)
                output = depth_decoder(output)
                


                pred_disp, _ = disp_to_depth(output[("disp", 0)],opt.min_depth, opt.max_depth)

                cam_points, _ = backproject_depth(pred_disp, invK)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                temp_k = invK.cpu().numpy()
                

                test = cam_points.cpu()[0].numpy()
                value_x = test[0,:]
                value_y = test[1,:]
                value_z = test[2,:]


                # points_array = np.array([[value_x[0], value_y[0], value_z[0]]], dtype=np.float32)
                # for k in range(len(value_x)-1):
                #     points_array = np.append(points_array, [[value_x[k], value_y[k], value_z[k]]], axis=0)
                streamarray=[]
                streamarray.append(value_x)
                streamarray.append(value_y)
                streamarray.append(value_z)

                streamarray = np.concatenate(streamarray)
                streamarray = np.reshape(streamarray, (3,-1))
                streamarray = np.transpose(streamarray)

                stream.append(streamarray)
                pred_disps.append(pred_disp)
                inv_ks.append(temp_k)
        pred_disps = np.concatenate(pred_disps)
        inv_ks = np.concatenate(inv_ks)
        print('finished predicting!')

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #delay = round(1000/30.0)
    #out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1216, 352))
    # for idx in range(len(pred_disps)):
    #     disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
    #     depth = np.clip(disp_resized, 0, 10)
    #     dmax, dmin = depth.max(), depth.min()
    #     depth = (depth)/(11)
    #     depth = np.uint8(depth * 256)
    #     #out.write(dtest = cam_pointsepth)
    #     cv2.imshow("test", depth)
    #     # filename11 = "./image_cmt/" +str(idx).zfill(3) +".png"
    #     # cv2.imwrite(filename11, depth)
    #     cv2.waitKey(33)



def inference():
    global pcd
    global pred_disps
    global counter
    global g_height
    global g_width
    global inv_ks
    global stream
    
    if len(stream) > counter:        
        pcd.points = o3d.utility.Vector3dVector(stream[counter])
        vis.update_geometry(pcd)
        time.sleep(0.03)
    else:
        counter = 0

    vis.update_geometry(pcd)
    counter = counter+1

def animation_callback(vis):  
    inference()


pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.VisualizerWithKeyCallback()
counter = 0

axis_pt = np.array([ [0, 0, 0], [0, -1, 0], [0,-1, 2],[0,0, 2], 
                     [1, 0, 0], [1, -1, 0], [1,-1, 2],[1, 0, 2],
                     [2, 0, 0], [2, -1, 0], [2,-1, 2],[2, 0, 2],
                     [3, 0, 0], [3, -1, 0], [3,-1, 2],[3, 0, 2],
                     [4, 0, 0], [4, -1, 0], [4,-1, 2],[4, 0, 2],[5,0,0],[0,-5,0],[0,0,5]], dtype=np.float32)

axis_lines = [[0,1], [1,2], [2,3], [3,0], 
                [4,5],[5,6],[6,7],[7,4],
                [8,9],[9,10],[10,11],[11,8],
                [12,13],[13,14],[14,15],[15,12],
                [16,17],[17,18],[18,19],[19,16], [0,16], [1,17], [0,20],[0,21],[0,22]]  

axis_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(axis_pt),
    lines=o3d.utility.Vector2iVector(axis_lines),
)


if __name__ == "__main__":
    options = MonodepthOptions()

    evaluate(options.parse())    
    vis.create_window()    
    vis.add_geometry(pcd)    
    vis.add_geometry(axis_line_set)    
    vis.register_animation_callback(animation_callback)
    vis.run()

