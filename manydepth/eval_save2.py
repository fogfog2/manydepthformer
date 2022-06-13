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

splits_dir = "/home/sj/src/manydepthformer/splits"

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

input_images = []
pred_disps = []
inv_ks = []
poses = []
ks = []
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
    global ks
    global input_images
    global poses

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    device = torch.device("cuda")
    
    cuda_device = "cuda:"+str(opt.cuda_device)
    
    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weightopts_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # Setup dataloaders
        
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files+.txt"))
        #filenames = readlines(os.path.join(splits_dir, "odom/test_files_09.txt"))
        #filenames = readlines(os.path.join(splits_dir, "odom/test_files_09.txt"))
        #filenames = readlines("/home/sj/colon_syn/test_files.txt")
        #filenames = readlines("/home/sj/colon/test_files_3.txt")
        #filenames = readlines("/media/sj/data/colon/images/images/test_files.txt")
        #filenames = filenames[::2]
        if opt.eval_teacher:
            encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
            encoder_class = networks.ResnetEncoder

        else:
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

            encoder_model = opt.train_model
            #encoder_model = "swin_h" 
            #encoder_model = "cmt_h"
            
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
                                               encoder_dict['height'], encoder_dict['width'],
                                               frames_to_load, 4,
                                               is_train=False,
                                               img_ext=img_ext)

        else:
            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               frames_to_load, 4,
                                               is_train=False,
                                               img_ext=img_ext)
 
            
            # dataset = datasets.KITTIOdomDataset(opt.data_path, filenames,
            #                                    encoder_dict['height'], encoder_dict['width'],
            #                                    frames_to_load, 4,
            #                                    is_train=False,
            #                                    img_ext=img_ext)

        
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

        # backproject_depth = BackprojectDepth(1, g_height, g_width)
        # backproject_depth.to("cuda")
        

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
                #input_image = data[('raw_color', 0, 0)]

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
                

                K = data[('K', 0)]  # quarter resolution for matching
                invK = data[('inv_K', 0)]

                _, pred_disp= disp_to_depth(output[("disp", 0)],opt.min_depth, opt.max_depth)

                
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                temp_invk = invK.cpu().numpy()                
                temp_k = K.cpu().numpy()         
                rgb = input_color.cpu().numpy()       
                pred_disps.append(pred_disp)
                inv_ks.append(temp_invk)
                ks.append(temp_k)
                input_images.append(rgb)
                pose= relative_poses.cpu()[0].numpy()
                poses.append(pose)

        pred_disps = np.concatenate(pred_disps)
        inv_ks = np.concatenate(inv_ks)
        ks = np.concatenate(ks)
        input_images = np.concatenate(input_images)
        poses = np.concatenate(poses)
        print('finished predicting!')

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            pred_disps = pred_disps[eigen_to_benchmark_ids]

    
    
    for idx in range(len(pred_disps)):
        disp_resized = cv2.resize(pred_disps[idx], (256, 256))        
        depth = np.clip(disp_resized, 0, 10)
        dmax, dmin = depth.max(), depth.min()
        size = dmax-dmin
        depth = (depth-dmin)/size
       # depth = 1-depth
        depth = np.uint8(depth * 255)
        #out.write(dtest = cam_pointsepth)
        cv2.imshow("test", depth)
        
        filename11 = "./image_cmt/" +str(idx).zfill(3) +".png"
        os.makedirs("image_cmt", exist_ok=True)
        depth = cv2.applyColorMap(depth , cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(filename11, depth)
        
        
        
        filename12 = "./image_cmt_rgb/" +str(idx).zfill(3) +".png"
        os.makedirs("image_cmt_rgb", exist_ok=True)
        image = np.uint8(input_images[idx]*255)
        # rgb = np.dstack((image[2],image[1],image[0]))
        #rgb_resized = cv2.resize(image[0], (640, 192))      
        
        #image = np.rollaxis(image,axis=0 , start=2)
        image = image.transpose(1,2,0)
        image = image[:,:, ::-1]
        cv2.imwrite(filename12, image)


        # filename12 = "./image_cmt_rgb/" +str(idx).zfill(3) +".png"
        # os.makedirs("image_cmt_rgb", exist_ok=True)
        # image = np.uint8(input_images[idx]*255)
        # rgb = np.dstack((image[2],image[1],image[0]))
        # rgb_resized = cv2.resize(rgb, (640, 192))      
        # cv2.imwrite(filename12, rgb_resized)




        #save 
        k= ks[idx]
        invk= inv_ks[idx]
        os.makedirs("image_cmt_intrinsic", exist_ok=True)
        file = open("./image_cmt_intrinsic/"+str(idx).zfill(3)+ ".txt","w")
        
        for i in range(len(invk)):
            for j in range(len(invk[i])):
                file.write(str(invk[i][j])+", ")
        file.write("\n")
        for i in range(len(k)):
            for j in range(len(k[i])):
                file.write(str(k[i][j])+", ")
        file.close()


        pose= poses[idx]
        os.makedirs("image_cmt_pose", exist_ok=True)
        file = open("./image_cmt_pose/"+str(idx).zfill(3)+ ".txt","w")
        for i in range(len(pose)):
            for j in range(len(pose[i])):
                file.write(str(pose[i][j])+", ")
        file.close()



if __name__ == "__main__":
    options = MonodepthOptions()

    evaluate(options.parse())    