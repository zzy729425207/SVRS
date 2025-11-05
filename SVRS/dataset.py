import os
import numpy as np
import skimage.measure
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])


import re


def file2caliblist(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    p_left = re.search(
        r'P_rect_101:\s*([\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+\s+[\d\.e+\-]+)',
        content)
    p_left_data = p_left.group(1).split()
    fx = float(p_left_data[0])
    fy = float(p_left_data[5])
    cx = float(p_left_data[2])
    cy = float(p_left_data[6])

    # 提取基线距离
    t_right = re.search(r'T_103:\s*([\d\.e+\-]+)', content)
    baseline = abs(float(t_right.group(1)))

    image_res = re.search(r'S_rect_101:\s*([\d\.e+\-]+)\s+([\d\.e+\-]+)', content)
    width = int(float(image_res.group(1)))
    height = int(float(image_res.group(2)))

    return [fx, fy, cx, cy, baseline, width, height]


def read_calib(calib_path):
    calib_file_list = os.listdir(calib_path)
    calib_table = {}
    for name in calib_file_list:
        calib_file = os.path.join(calib_path, name)
        calib_name = name.split(".")[0]
        calib_table[f"{calib_name}"] = file2caliblist(calib_file)
    return calib_table


class VoxelDSDataset(Dataset):

    def __init__(self, datapath, list_filename, calib_path, training, transform=True, lite=False):  # 是否训练，是否轻量
        self.datapath = datapath
        self.calib_path = calib_path
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None
        self.calib = read_calib(calib_path)

        # Camera intrinsics
        # 15mm images have different focals
        self.voxel_size = 0.2
        # self.voxel_size = [1.6,0.8,0.4]
        # self.grid_sizes = [32, 64,128]
        self.grid_sizes = [16, 32, 64, 128]
        self.voxel_scale = 4
        # set the maximum perception depth
        self.max_depth = self.voxel_size * self.grid_sizes[-1]
        self.transform = transform
        self.max_disp = 192

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def project_image_to_rect(self, uv_depth, fu, fv, cu, cv, baseline):
        ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - cu) * uv_depth[:, 2]) / \
            fu + baseline
        y = ((uv_depth[:, 1] - cv) * uv_depth[:, 2]) / fv
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def calc_cloud(self, disp_est, depth, fu, fv, cu, cv, baseline):
        mask = disp_est > 0
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]
        cloud = self.project_image_to_rect(points, fu, fv, cu, cv, baseline)
        return cloud

    def filter_cloud(self, cloud):
        min_mask = cloud >= [-(self.max_depth / 2), -(self.max_depth / 2), 0.0]
        max_mask = cloud <= [(self.max_depth / 2), (self.max_depth / 2), self.max_depth]
        min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
        max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
        filter_mask = min_mask & max_mask
        filtered_cloud = cloud[filter_mask]
        return filtered_cloud

    def calc_voxel_grid(self, filtered_cloud, grid_size):
        voxel_size = self.max_depth / grid_size
        # quantized point values, here you will loose precision
        xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
        # Empty voxel grid
        vox_grid = np.zeros((grid_size, grid_size, grid_size))  # 空的栅格地图

        # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
        offsets = np.array([int(grid_size / 2), int(grid_size / 2), 0])
        xyz_offset_q = xyz_q + offsets

        # Setting all voxels containitn a points equal to 1
        vox_grid[xyz_offset_q[:, 0],
                 xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1

        # get back indexes of populated voxels
        xyz_v = np.asarray(np.where(vox_grid == 1))
        cloud_np = np.asarray([(pt - offsets) * voxel_size for pt in xyz_v.T])
        return vox_grid, cloud_np

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        #  加载图片和视差
        left_img = self.load_image(os.path.join(
            self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(
            self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(
            self.datapath, self.disp_filenames[index]))

        calib_name = self.left_filenames[index].split("/")[1]
        calib_data = self.calib[calib_name]
        fu = calib_data[0]
        fv = calib_data[1]
        cu = calib_data[2]
        cv = calib_data[3]
        baseline = calib_data[4]

        #vox_cost_vol_disp_set = set()
        #for z in np.arange(self.voxel_size * self.voxel_scale, self.max_depth, self.voxel_size * self.voxel_scale):
        #    d = fu * baseline / z
        #    if d > self.max_disp:
        #        continue
        #    vox_cost_vol_disp_set.add(round(d / self.voxel_scale))
        #vox_cost_vol_disps = list(vox_cost_vol_disp_set)
        #vox_cost_vol_disps = sorted(vox_cost_vol_disps)
        
        vox_cost_vol_disps = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 28, 34, 42]


        w, h = left_img.size
        crop_w, crop_h = 880, 400

        processed = get_transform()

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img).numpy()
                right_img = processed(right_img).numpy()

                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, crop_w - w)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, crop_w - w)), mode='constant', constant_values=0)
                disparity = np.lib.pad(
                    disparity, ((0, 0), (0, crop_w - w)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
                right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
                disparity = disparity[h - crop_h:h, w - crop_w: w]

                left_img = processed(left_img)
                right_img = processed(right_img)
        else:
            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)

        # 将视差图转为深度图
        mask = disparity > 0
        depth_gt = fu * baseline / (disparity + 1. - mask)

        # 点云真值
        cloud_gt = self.calc_cloud(disparity, depth_gt, fu, fv, cu, cv, baseline)

        # 对点云进行过滤
        filtered_cloud_gt = self.filter_cloud(cloud_gt)

        # 体素网格真值  [8,16,32,64]
        all_vox_grid_gt = []
        for grid_size in self.grid_sizes:
            # 将过滤后的点云按照网格尺寸参数进行网格化
            vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(filtered_cloud_gt, grid_size=grid_size)
            vox_grid_gt = torch.from_numpy(vox_grid_gt)
            all_vox_grid_gt.append(vox_grid_gt)

        return {"left": left_img,  # 左图
                "right": right_img,  # 右图
                "disparity": disparity,  # 视差图
                "voxel_grid": all_vox_grid_gt,  # 体素图
                "vox_cost_vol_disps": vox_cost_vol_disps,  # 体素代价体
                "top_pad": 0,
                "right_pad": 0,
                "left_filename": self.left_filenames[index]}


class VoxelTestDSDataset(Dataset):

    def __init__(self, datapath, list_filename, training, transform=True, lite=False):  # 是否训练，是否轻量
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

        # Camera intrinsics
        # 15mm images have different focals
        self.voxel_size = 0.2
        # self.voxel_size = [1.6,0.8,0.4]
        # self.grid_sizes = [32, 64,128]
        self.grid_sizes = [16, 32, 64, 128]
        self.voxel_scale = 4
        # set the maximum perception depth
        self.max_depth = self.voxel_size * self.grid_sizes[-1]
        self.transform = transform
        self.max_disp = 192

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def project_image_to_rect(self, uv_depth, fu, fv, cu, cv, baseline):
        ''' Input: nx3 first two channels are uv, 3rd channel
                is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - cu) * uv_depth[:, 2]) / \
            fu + baseline
        y = ((uv_depth[:, 1] - cv) * uv_depth[:, 2]) / fv
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def calc_cloud(self, disp_est, depth, fu, fv, cu, cv, baseline):
        mask = disp_est > 0
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth])
        points = points.reshape((3, -1))
        points = points.T
        points = points[mask.reshape(-1)]
        cloud = self.project_image_to_rect(points, fu, fv, cu, cv, baseline)
        return cloud

    def filter_cloud(self, cloud):
        min_mask = cloud >= [-(self.max_depth / 2), -(self.max_depth / 2), 0.0]
        max_mask = cloud <= [(self.max_depth / 2), (self.max_depth / 2), self.max_depth]
        min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
        max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
        filter_mask = min_mask & max_mask
        filtered_cloud = cloud[filter_mask]
        return filtered_cloud

    def calc_voxel_grid(self, filtered_cloud, grid_size):
        voxel_size = self.max_depth / grid_size
        # quantized point values, here you will loose precision
        xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
        # Empty voxel grid
        vox_grid = np.zeros((grid_size, grid_size, grid_size))  # 空的栅格地图

        # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
        offsets = np.array([int(grid_size / 2), int(grid_size / 2), 0])
        xyz_offset_q = xyz_q + offsets

        # Setting all voxels containitn a points equal to 1
        vox_grid[xyz_offset_q[:, 0],
                 xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1

        # get back indexes of populated voxels
        xyz_v = np.asarray(np.where(vox_grid == 1))
        cloud_np = np.asarray([(pt - offsets) * voxel_size for pt in xyz_v.T])
        return vox_grid, cloud_np

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):

        #  加载图片和视差
        left_img = self.load_image(os.path.join(
            self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(
            self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(
            self.datapath, self.disp_filenames[index]))

        #calib_name = self.left_filenames[index].split("/")[1]
        #calib_data = self.calib[calib_name]
        #fu = calib_data[0]
        #fv = calib_data[1]
        #cu = calib_data[2]
        #cv = calib_data[3]
        #baseline = calib_data[4].
        cu = 4.556890e+2
        cv = 1.976634e+2
        fu = 1.003556e+3
        fv = 1.003556e+3
        baseline = 0.54
        
        

        #vox_cost_vol_disp_set = set()
        #for z in np.arange(self.voxel_size * self.voxel_scale, self.max_depth, self.voxel_size * self.voxel_scale):
        #    d = fu * baseline / z
        #    if d > self.max_disp:
        #        continue
        #    vox_cost_vol_disp_set.add(round(d / self.voxel_scale))
        #vox_cost_vol_disps = list(vox_cost_vol_disp_set)
        #vox_cost_vol_disps = sorted(vox_cost_vol_disps)
        
        vox_cost_vol_disps = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 21, 24, 28, 34, 42]


        w, h = left_img.size
        crop_w, crop_h = 880, 400

        processed = get_transform()

        if self.transform:
            if w < crop_w:
                left_img = processed(left_img).numpy()
                right_img = processed(right_img).numpy()

                left_img = np.lib.pad(
                    left_img, ((0, 0), (0, 0), (0, crop_w - w)), mode='constant', constant_values=0)
                right_img = np.lib.pad(
                    right_img, ((0, 0), (0, 0), (0, crop_w - w)), mode='constant', constant_values=0)
                disparity = np.lib.pad(
                    disparity, ((0, 0), (0, crop_w - w)), mode='constant', constant_values=0)

                left_img = torch.Tensor(left_img)
                right_img = torch.Tensor(right_img)
            else:
                left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
                right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
                disparity = disparity[h - crop_h:h, w - crop_w: w]

                left_img = processed(left_img)
                right_img = processed(right_img)
        else:
            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            left_img = np.asarray(left_img)
            right_img = np.asarray(right_img)

        # 将视差图转为深度图
        mask = disparity > 0
        depth_gt = fu * baseline / (disparity + 1. - mask)

        # 点云真值
        cloud_gt = self.calc_cloud(disparity, depth_gt, fu, fv, cu, cv, baseline)

        # 对点云进行过滤
        filtered_cloud_gt = self.filter_cloud(cloud_gt)

        # 体素网格真值  [8,16,32,64]
        all_vox_grid_gt = []
        for grid_size in self.grid_sizes:
            # 将过滤后的点云按照网格尺寸参数进行网格化
            vox_grid_gt, cloud_np_gt = self.calc_voxel_grid(filtered_cloud_gt, grid_size=grid_size)
            vox_grid_gt = torch.from_numpy(vox_grid_gt)
            all_vox_grid_gt.append(vox_grid_gt)

        return {"left": left_img,  # 左图
                "right": right_img,  # 右图
                "disparity": disparity,  # 视差图
                "voxel_grid": all_vox_grid_gt,  # 体素图
                "vox_cost_vol_disps": vox_cost_vol_disps,  # 体素代价体
                "top_pad": 0,
                "right_pad": 0,
                "left_filename": self.left_filenames[index]}
