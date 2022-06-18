# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import csv
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
from manydepth.layers import transformation_from_parameters, disp_to_depth
import tqdm
import pandas as pd
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = "splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    
    if opt.cuda_device is None:
        cuda_device = "cuda:0"
    else:
        cuda_device = opt.cuda_device

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # Setup dataloaders
        assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
             "eval_split should be either odom_9 or odom_10"

        sequence_id = int(opt.eval_split.split("_")[1])

        filenames = readlines(
            os.path.join(splits_dir, "odom",
                        "test_files_{:02d}.txt".format(sequence_id)))

        
       # filenames = filenames[::2]
        if opt.eval_teacher:
            encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
            encoder_class = networks.ResnetEncoder

        else:
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

            #encoder_model = "resnet" 
            #encoder_model = "swin_h" 
            encoder_model =  opt.train_model
            
            if "resnet" in encoder_model:            
                encoder_class = networks.ResnetEncoderMatching
            elif "swin" in encoder_model:
                encoder_class = networks.SwinEncoderMatching
            elif "cmt" in encoder_model:
                encoder_class = networks.CMTEncoderMatching
                    
            #encoder_class = networks.ResnetEncoderMatching

        encoder_dict = torch.load(encoder_path)
        try:
            HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
        except KeyError:
            print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                  'using command line values!')
            HEIGHT, WIDTH = opt.height, opt.width

        img_ext = '.png' if opt.png else '.jpg'
        if opt.eval_split == 'cityscapes':
            dataset = datasets.CityscapesEvalDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False,
                                                     img_ext=img_ext)
        elif opt.eval_split =='custom_ucl':
            dataset = datasets.CustomUCLRAWDataset(opt.data_path, filenames,
                                                     HEIGHT, WIDTH,
                                                     frames_to_load, 4,
                                                     is_train=False,
                                                     img_ext=img_ext)
        else:
            dataset = datasets.KITTIOdomDataset(opt.data_path, filenames,
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
            
            if "resnet" in encoder_model:            
                encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins)
            elif "swin" in encoder_model:
                encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins, use_swin_feature = opt.swin_use_feature)
            elif "cmt" in encoder_model:
                encoder_opts = dict(num_layers=opt.num_layers,
                                pretrained=False,
                                input_width=encoder_dict['width'],
                                input_height=encoder_dict['height'],
                                adaptive_bins=True,
                                min_depth_bin=0.1, max_depth_bin=20.0,
                                depth_binning=opt.depth_binning,
                                num_depth_bins=opt.num_depth_bins,
                                upconv = opt.cmt_use_upconv, start_layer = opt.cmt_layer, embed_dim = opt.cmt_dim,  use_cmt_feature = opt.cmt_use_feature
                                )
            
            
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
                pose_enc.cuda(cuda_device)
                pose_dec.cuda(cuda_device)

        encoder = encoder_class(**encoder_opts)                
        
        
        if opt.use_attention_decoder:            
            depth_decoder = networks.DepthDecoderAttention(encoder.num_ch_enc , no_spatial= opt.attention_only_channel)           
        else:
            depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.eval()
        depth_decoder.eval()

        if torch.cuda.is_available():
            encoder.cuda(cuda_device)
            depth_decoder.cuda(cuda_device)

        pred_disps = []
        pred_poses = []

        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color', 0, 0)]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()

                if opt.eval_teacher:
                    output = encoder(input_color)
                    output = depth_decoder(output)
                else:

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
                                axisangle[:, 0], translation[:, 0])

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

 
                    pred_poses.append(pose.cpu().numpy())

        #red_disps = np.concatenate(pred_disps)
        pred_poses = np.concatenate(pred_poses)
        print('finished predicting!')

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]



    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]
    #gt_xyzs = gt_global_poses[30:, :3, 3]
    #gt_global_poses = gt_global_poses[30:]
    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    #all_pred =[]
    
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        #all_pred.append(local_xyzs)
        
    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))


    all_pred= np.array(dump_xyz(pred_poses[0:num_frames - 1]))
    all_gts= np.array(dump_xyz(gt_local_poses[0:num_frames- 1]))
    
    offset = all_gts[0] - all_pred[0]
    pred_xyz = all_pred + offset[None, :]

    # Optimize the scaling factor
    
    scale = np.sum(all_gts * pred_xyz) / np.sum(pred_xyz ** 2)
    
    pred_result = pred_xyz * scale 
    all_result = all_gts
    
    pd.DataFrame(ates).to_csv("pred_10_rmse.csv")
    pd.DataFrame(pred_result).to_csv("pred_10_s.csv")
    #pd.DataFrame(all_result).to_csv("gt_10_s.csv")
    
    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
