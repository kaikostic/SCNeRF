import os
import numpy as np
import imageio
import logging
from nerf_sample_ray_split import RaySamplerSingleImage
import glob

import torch

logger = logging.getLogger(__package__)

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []


def load_data_split(basedir, scene, split, args, skip=1, dscale=2, try_load_min_depth=True, only_img_files=False):

    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        if len(nums) == 18: 
            return np.array(
                [float(x) for x in nums[:16]]
            ).reshape([4, 4]).astype(np.float32), np.array(
                [float(x) for x in nums[16:]]
            ).astype(np.float32)
        else:
            assert len(nums) == 16
            return np.array(
                [float(x) for x in nums]
            ).reshape([4, 4]).astype(np.float32), None

    if basedir[-1] == '/':          # remove trailing '/'
        basedir = basedir[:-1]
     
    split_dir = '{}/{}/{}'.format(basedir, scene, split)
    if only_img_files:
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        return img_files

    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))
    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    cam_cnt = len(pose_files)
    #print('intrinsics_files',len(intrinsics_files))
    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        logger.info('raw img_files: {}'.format(len(img_files)))
        img_files = img_files[::skip]
        assert(len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt

    # mask files
    mask_files = find_files('{}/mask'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(mask_files) > 0:
        logger.info('raw mask_files: {}'.format(len(mask_files)))
        mask_files = mask_files[::skip]
        assert(len(mask_files) == cam_cnt)
    else:
        mask_files = [None, ] * cam_cnt

    # min depth files
    mindepth_files = find_files('{}/min_depth'.format(split_dir), exts=['*.png', '*.jpg'])
    if try_load_min_depth and len(mindepth_files) > 0:
        logger.info('raw mindepth_files: {}'.format(len(mindepth_files)))
        mindepth_files = mindepth_files[::skip]
        assert(len(mindepth_files) == cam_cnt)
    else:
        mindepth_files = [None, ] * cam_cnt

    # assume all images have the same size as training image
    #resize
    train_imgfile = find_files('{}/{}/train/rgb'.format(basedir, scene), exts=['*.png', '*.jpg'])[0]
    train_im = imageio.imread(train_imgfile)
    H_, W_ = train_im.shape[:2]
    H = H_ // dscale
    W = W_ // dscale

    # create ray samplers
    ray_samplers = []
    extrinsics = []
    #print('imgfiles',img_files)
    for i in range(cam_cnt):
        intrinsics, k = parse_txt(intrinsics_files[i])
        #Downscale intrinsics
        intrinsics[0,0]=intrinsics[0,0] * W / W_ #fx
        intrinsics[1,1]=intrinsics[1,1] * H / H_ #fy
        intrinsics[0,2]=intrinsics[0,2] * W / W_ #cx
        intrinsics[1,2]=intrinsics[1,2] * H / H_ #cy 
        pose, _ = parse_txt(pose_files[i])
        pose[:3, 3] /= args.normalize_factor
        extrinsics.append(pose)
        # read max depth
        try:
            max_depth = float(open('{}/max_depth.txt'.format(split_dir)).readline().strip())
        except:
            max_depth = None
        
        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose, k=k,
                                                  img_path=img_files[i],
                                                  mask_path=mask_files[i],
                                                  min_depth_path=mindepth_files[i],
                                                  max_depth=max_depth))
        #print(ray_samplers)
    logger.info('Split {}, # views: {}'.format(split, cam_cnt))

    return ray_samplers, {
        "intrinsics": torch.from_numpy(intrinsics),
        "extrinsics": torch.tensor(extrinsics),
        "H": H, "W": W, "k": k
    }
