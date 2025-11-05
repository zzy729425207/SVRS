import time
import torch
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt



def load_image(filename):
    return Image.open(filename).convert('RGB')


def load_disp(filename):
    data = Image.open(filename)
    data = np.array(data, dtype=np.float32) / 256.
    return data


def project_image_to_rect(uv_depth):
    ''' Input: nx3 first two channels are uv, 3rd channel
               is depth in rect camera coord.
        Output: nx3 points in rect camera coord.
    '''
    c_u = 4.556890e+2
    c_v = 1.976634e+2
    f_u = 1.003556e+3
    f_v = 1.003556e+3
    b_x = 0.0
    b_y = 0.0
    # c_u = 1057.288
    # c_v = 1057.288
    # f_u = 1129.229
    # f_v = 1057.288
    # b_x = 0.0
    # b_y = 0.0
    n = uv_depth.shape[0]
    x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
    y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
    pts_3d_rect = np.zeros((n, 3))
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = uv_depth[:, 2]
    return pts_3d_rect


def Sobel_Mask(image, th=120):
    sobel_car1 = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    sobel_car2 = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    sobel_car1 = cv2.convertScaleAbs(sobel_car1)  # 转回uint8
    sobel_car2 = cv2.convertScaleAbs(sobel_car2)

    sobel_car = cv2.addWeighted(sobel_car1, 0.5, sobel_car2, 0.5, 0) > th
    return sobel_car


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

def Depth_GT(Vis=True,
             left_Path="./demo/left1.jpg",
             disparity_Path="./demo/disparity1.png",
             Save_index="GT", voxel_size=0.4, voxel_range=[32, 32, 32], eav=False):
    # 读取图片
    if not eav:
        print("读取图片！")
    left_img = load_image(left_Path)
    disparity = load_disp(disparity_Path)

    # 处理图片
    if not eav:
        print("处理图片！")
    w, h = left_img.size
    crop_w, crop_h = 880, 400
    disparity = disparity[h - crop_h:h, w - crop_w: w]

    # 将视差点进行过滤  并  转为三维点云
    mask = disparity > 0  # 二维矩阵，元素是False 或者 Ture
    f_u = 1.003556e+3
    baseline = 0.54
    depth_gt = f_u * baseline / (disparity + 1. - mask)
    rows, cols = depth_gt.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth_gt])
    points = points.reshape((3, -1))  # 3,352000
    points = points.T
    points = points[mask.reshape(-1)]  # 过滤掉废弃点  65847,3 从352000过滤后剩余65847
    cloud_gt = project_image_to_rect(points)  # 65847,3 转为3维点，从归一化的矩阵网格转为三维

    # 将点云网格化
    min_mask = cloud_gt >= [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0]  # 2维的
    max_mask = cloud_gt <= [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size,
                            voxel_range[1] * voxel_size]
    if not eav:
        print("最小取值范围:", [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0], "\t", "最大取值范围:",
              [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size, voxel_range[1] * voxel_size])
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask  # 成为一个一维的
    filtered_cloud = cloud_gt[filter_mask]  # 全部过滤，速度更快  将体素网格以外的点进行过滤 过滤后剩余5966个点
    if not eav:
        print("共有{}个点\t其中有{}个点有效且在范围内\t雷达点云有效率为{}%。".format(points.shape[0], filtered_cloud.shape[0],
                                                            round((filtered_cloud.shape[0] / points.shape[0]) * 100,
                                                                  2)))

    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
    # vox_grid = np.zeros((int(32 / voxel_size), int(int(4 + H_mask) / voxel_size), int(24 / voxel_size)))  # 空的栅格地图
    vox_grid = np.zeros((int(voxel_range[0]), int(voxel_range[2]), int(voxel_range[1])))  # 空的栅格地图

    offsets = np.array([int(voxel_range[0] / 2), int(voxel_range[2] / 2), 0])  # 偏移量  与图片对齐，方便后续与图片的联合可视化
    xyz_offset_q = xyz_q + offsets  # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1  # 将空栅格通过索引赋值1 大小(256, 256, 32)
    if not eav:
        print("当前帧：深度图==》》Voxel成功！！！")

    return filtered_cloud, vox_grid

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def Depth_RAFT(Vis=True,
               left_Path=r"F:\DataSets\DrivingStereo\test_left\2018-08-13-15-32-19\2018-08-13-15-32-19_2018-08-13-15-32-52-955.jpg",
               right_Path=r"F:\DataSets\DrivingStereo\test_right\2018-08-13-15-32-19\2018-08-13-15-32-19_2018-08-13-15-32-52-955.jpg",
               Save_index="RAFT", voxel_size=0.4, voxel_range=[32, 32, 32], eva=False, Mask_edge=True):
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'],
                        help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()
    # 加载模型
    from RAFT.raft_stereo import RAFTStereo
    model = RAFTStereo(args).cuda()  # 立体匹配
    model = nn.DataParallel(model)
    model.eval()
    model.load_state_dict(torch.load("./RAFT/iraftstereo_rvc.pth"))
    if not eva:
        print("INFO  ==>>  RAFT Loading Success!")

        # 读取图片
        print("读取图片！")
    left_img = load_image(left_Path)
    right_img = load_image(right_Path)

    # 处理图片
    if not eva:
        print("处理图片！")
    w, h = left_img.size
    crop_w, crop_h = 880, 400
    left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
    if Mask_edge:
        sobel_mask = Sobel_Mask(cv2.cvtColor(np.asarray(left_img), cv2.COLOR_RGB2GRAY), th=10)

    right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
    processed = get_transform()
    left_img = torch.unsqueeze(processed(left_img), 0)
    right_img = torch.unsqueeze(processed(right_img), 0)
    time1 = time.time()
    with torch.no_grad():
        disparity_list = model(left_img.cuda(), right_img.cuda())[22:]
        if not eva:
            print("共有{}次循环参与飞点过滤".format(len(disparity_list)))
        for i in range(len(disparity_list)):
            if i != 0:
                if i == 1:
                    disparity_diff = torch.abs(disparity_list[i] - disparity_list[i - 1])
                    disparity_mean = torch.mean(disparity_diff)
                    bad_mask = disparity_diff > disparity_mean
                else:
                    disparity_diff = torch.abs(disparity_list[i] - disparity_list[i - 1])
                    disparity_mean = torch.mean(disparity_diff)
                    bad_mask &= disparity_diff > disparity_mean
        disparity = disparity_list[-1]
        disparity[bad_mask] = 0
        disparity = -torch.squeeze(disparity)
        if Mask_edge:
            disparity[sobel_mask] = 0
        disp_est_np = disparity.data.cpu().numpy()
        disp_est = np.array(disp_est_np, dtype=np.float32)
        disp_est[disp_est < 0] = 0
    if not eva:
        print("预测视差完成，耗时_{}！".format(time.time() - time1))

    # 将视差点进行过滤  并  转为三维点云
    f_u = 1.003556e+3
    baseline = 0.54
    # f_u = 1.057288e+3
    # baseline = 0.12

    mask = disp_est > 0
    depth_gt = f_u * baseline / (disp_est + 1. - mask)
    rows, cols = depth_gt.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth_gt])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud_gt = project_image_to_rect(points)

    # 将点云网格化
    min_mask = cloud_gt >= [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0]  # 2维的
    max_mask = cloud_gt <= [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size,
                            voxel_range[1] * voxel_size]
    if not eva:
        print("最小取值范围:", [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0], "\t", "最大取值范围:",
              [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size, voxel_range[1] * voxel_size])
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask  # 成为一个一维的
    filtered_cloud = cloud_gt[filter_mask]  # 全部过滤，速度更快  将体素网格以外的点进行过滤 过滤后剩余5966个点
    if not eva:
        print("共有{}个点\t其中有{}个点有效且在范围内\t雷达点云有效率为{}%。".format(points.shape[0], filtered_cloud.shape[0],
                                                            round((filtered_cloud.shape[0] / points.shape[0]) * 100,
                                                                  2)))

    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
    # vox_grid = np.zeros((int(32 / voxel_size), int(int(4 + H_mask) / voxel_size), int(24 / voxel_size)))  # 空的栅格地图
    vox_grid = np.zeros((int(voxel_range[0]), int(voxel_range[2]), int(voxel_range[1])))  # 空的栅格地图

    offsets = np.array([int(voxel_range[0] / 2), int(voxel_range[2] / 2), 0])  # 偏移量  与图片对齐，方便后续与图片的联合可视化
    xyz_offset_q = xyz_q + offsets  # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1  # 将空栅格通过索引赋值1 大小(256, 256, 32)
    if not eva:
        print("当前帧：深度图==》》Voxel成功！！！")

    return filtered_cloud, vox_grid


def Depth_TTF_P(Vis=True,
                left_Path=r"F:\桌面文件\障碍物检测\障碍物检测程序\demo\left1.jpg",
                right_Path=r"F:\桌面文件\障碍物检测\障碍物检测程序\demo\right1.jpg",
                Save_index="TTF_P", Mask_Edge=True, Mask_FlyPoints=True, voxel_size=0.4, voxel_range=[32, 32, 32],
                eva=False):
    from TTFStereo_performance.TTFStereo import TTFStereo_performance
    model = TTFStereo_performance().cuda()  # 立体匹配
    model = nn.DataParallel(model)
    model.eval()
    # model.load_state_dict(torch.load("./TTFStereo_performance/TTFStereo_domain_1_4_16_add_iter16.pth"))
    model.load_state_dict(torch.load("./TTFStereo_performance/20_TTF_P_DS.pth"))
    if not eva:
        print("INFO  ==>>  TTF_P Loading Success!")

        # 读取图片
        print("读取图片！")
    left_img = load_image(left_Path)
    right_img = load_image(right_Path)

    # 处理图片
    if not eva:
        print("处理图片！")
    w, h = left_img.size
    crop_w, crop_h = 880, 400
    left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
    if Mask_Edge:
        sobel_mask = Sobel_Mask(cv2.cvtColor(np.asarray(left_img), cv2.COLOR_RGB2GRAY), th=10)
    right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
    processed = get_transform()
    left_img = torch.unsqueeze(processed(left_img), 0)
    right_img = torch.unsqueeze(processed(right_img), 0)
    time1 = time.time()

    padder = InputPadder(left_img.shape, divis_by=32)
    left_img, right_img = padder.pad(left_img, right_img)

    with torch.no_grad():
        _, disparity_list = model(left_img.cuda(), right_img.cuda())
        disparity = disparity_list[-1]
        # disparity = _
        if Mask_FlyPoints:
            if not eva:
                print("共有{}帧参与飞点过滤".format(len(disparity_list)))
            for i in range(len(disparity_list)):
                if i != 0:
                    if i == 1:
                        disparity_diff = torch.abs(disparity_list[i] - disparity_list[i - 1])
                        disparity_mean = torch.mean(disparity_diff)
                        bad_mask = disparity_diff > disparity_mean
                    else:
                        disparity_diff = torch.abs(disparity_list[i] - disparity_list[i - 1])
                        disparity_mean = torch.mean(disparity_diff)
                        bad_mask &= disparity_diff > disparity_mean
            disparity[bad_mask] = 0
        disparity = padder.unpad(disparity.float()).squeeze()
        # disparity = torch.squeeze(disparity_list[-1])
        if Mask_Edge:
            disparity[sobel_mask] = 0
        disp_est_np = disparity.data.cpu().numpy()
        disp_est = np.array(disp_est_np, dtype=np.float32)
        disp_est[disp_est < 0] = 0
    if not eva:
        print("预测视差完成，耗时_{}！".format(time.time() - time1))

    # 将视差点进行过滤  并  转为三维点云
    f_u = 1.003556e+3
    baseline = 0.54
    # f_u = 1.057288e+3
    # baseline = 0.12

    mask = disp_est > 0
    depth_gt = f_u * baseline / (disp_est + 1. - mask)
    rows, cols = depth_gt.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth_gt])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud_gt = project_image_to_rect(points)

    # 将点云网格化
    min_mask = cloud_gt >= [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0]  # 2维的
    max_mask = cloud_gt <= [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size,
                            voxel_range[1] * voxel_size]
    if not eva:
        print("最小取值范围:", [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0], "\t", "最大取值范围:",
              [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size, voxel_range[1] * voxel_size])
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask  # 成为一个一维的
    filtered_cloud = cloud_gt[filter_mask]  # 全部过滤，速度更快  将体素网格以外的点进行过滤 过滤后剩余5966个点
    if not eva:
        print("共有{}个点\t其中有{}个点有效且在范围内\t雷达点云有效率为{}%。".format(points.shape[0], filtered_cloud.shape[0],
                                                            round((filtered_cloud.shape[0] / points.shape[0]) * 100,
                                                                  2)))

    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
    # vox_grid = np.zeros((int(32 / voxel_size), int(int(4 + H_mask) / voxel_size), int(24 / voxel_size)))  # 空的栅格地图
    vox_grid = np.zeros((int(voxel_range[0]), int(voxel_range[2]), int(voxel_range[1])))  # 空的栅格地图

    offsets = np.array([int(voxel_range[0] / 2), int(voxel_range[2] / 2), 0])  # 偏移量  与图片对齐，方便后续与图片的联合可视化
    xyz_offset_q = xyz_q + offsets  # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1  # 将空栅格通过索引赋值1 大小(256, 256, 32)
    if not eva:
        print("当前帧：深度图==》》Voxel成功！！！")
        print("预测体素网格完成，总耗时_{}！".format(time.time() - time1))

    return cloud_gt, vox_grid


def Depth_BG(Vis=True,
             left_Path="./demo/left.jpg",
             right_Path="./demo/right.jpg",
             Save_index="BG+", Mask_Edge=True, Mask_FlyPoints=True, voxel_size=0.4, voxel_range=[32, 32, 32]):
    from BGNet.models.bgnet_plus import BGNet_Plus
    model = BGNet_Plus().cuda()  # 立体匹配
    model = nn.DataParallel(model)
    model.eval()
    model.load_state_dict(torch.load(r"BGNet/BGNet_Plus.pth"))
    # print("INFO  ==>>  BG+ Loading Success!")

    # 读取图片
    # print("读取图片！")
    left_img = load_image(left_Path)
    right_img = load_image(right_Path)

    # 处理图片
    # print("处理图片！")
    w, h = left_img.size
    crop_w, crop_h = 880, 400
    left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
    if Mask_Edge:
        sobel_mask = Sobel_Mask(cv2.cvtColor(np.asarray(left_img), cv2.COLOR_RGB2GRAY), th=10)
    if Vis:
        # 这部分仅用于可视化
        points_rgb = np.transpose(np.asarray(left_img)[:, :, :3], (2, 0, 1)).reshape((3, -1)).T
    right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
    processed = get_transform()
    left_img = torch.unsqueeze(processed(left_img), 0)
    right_img = torch.unsqueeze(processed(right_img), 0)
    time1 = time.time()

    padder = InputPadder(left_img.shape, divis_by=64)
    left_img, right_img = padder.pad(left_img, right_img)

    with torch.no_grad():
        disparity = model(left_img.cuda(), right_img.cuda())[-1]
        disparity = padder.unpad(disparity.float()).squeeze()
        # disparity = torch.squeeze(disparity_list[-1])
        if Mask_Edge:
            disparity[sobel_mask] = 0
        disp_est_np = disparity.data.cpu().numpy()
        disp_est = np.array(disp_est_np, dtype=np.float32)
        disp_est[disp_est < 0] = 0
    # print("预测视差完成，耗时_{}！".format(time.time() - time1))

    # 将视差点进行过滤  并  转为三维点云
    f_u = 1.003556e+3
    baseline = 0.54
    # f_u = 1.057288e+3
    # baseline = 0.12

    mask = disp_est > 0
    depth_gt = f_u * baseline / (disp_est + 1. - mask)
    rows, cols = depth_gt.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth_gt])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud_gt = project_image_to_rect(points)

    # 将点云网格化
    min_mask = cloud_gt >= [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0]  # 2维的
    max_mask = cloud_gt <= [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size,
                            voxel_range[1] * voxel_size]
    # print("最小取值范围:", [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0], "\t", "最大取值范围:",
    #      [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size, voxel_range[1] * voxel_size])
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask  # 成为一个一维的
    filtered_cloud = cloud_gt[filter_mask]  # 全部过滤，速度更快  将体素网格以外的点进行过滤 过滤后剩余5966个点
    # print("共有{}个点\t其中有{}个点有效且在范围内\t雷达点云有效率为{}%。".format(points.shape[0], filtered_cloud.shape[0],
    #                                                    round((filtered_cloud.shape[0] / points.shape[0]) * 100, 2)))

    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
    # vox_grid = np.zeros((int(32 / voxel_size), int(int(4 + H_mask) / voxel_size), int(24 / voxel_size)))  # 空的栅格地图
    vox_grid = np.zeros((int(voxel_range[0]), int(voxel_range[2]), int(voxel_range[1])))  # 空的栅格地图

    offsets = np.array([int(voxel_range[0] / 2), int(voxel_range[2] / 2), 0])  # 偏移量  与图片对齐，方便后续与图片的联合可视化
    xyz_offset_q = xyz_q + offsets  # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1  # 将空栅格通过索引赋值1 大小(256, 256, 32)
    # print("当前帧：深度图==》》Voxel成功！！！")
    # print("预测体素网格完成，总耗时_{}！".format(time.time() - time1))

    return cloud_gt, vox_grid


def Depth_SGM(Vis=True,
              left_Path="./demo/left2.jpg",
              right_Path="./demo/right2.jpg",
              Save_index="SGM", Mask_Edge=True, voxel_size=0.4, voxel_range=[32, 32, 32], eva=True):
    win_size = 5
    min_disp = 0
    max_disp = 192  # min_disp * 9
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    # Create Block matching object.
    sgbm = cv2.StereoSGBM_create(minDisparity=min_disp,
                                 numDisparities=num_disp,
                                 blockSize=5,
                                 uniquenessRatio=5,
                                 speckleWindowSize=5,
                                 speckleRange=5,
                                 disp12MaxDiff=1,
                                 P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                 P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)

    # 读取图片
    if not eva:
        print("读取图片！")
    left_img = load_image(left_Path)
    right_img = load_image(right_Path)

    # 处理图片
    if not eva:
        print("处理图片！")
    w, h = left_img.size
    crop_w, crop_h = 880, 400
    left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
    if Mask_Edge:
        sobel_mask = Sobel_Mask(cv2.cvtColor(np.asarray(left_img), cv2.COLOR_RGB2GRAY), th=10)
    right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
    time1 = time.time()
    sgbm_disparity = sgbm.compute(cv2.cvtColor(np.asarray(left_img), cv2.COLOR_BGR2GRAY),
                                  cv2.cvtColor(np.asarray(right_img), cv2.COLOR_BGR2GRAY))
    if not eva:
        print("预测视差完成，耗时_{}！".format(time.time() - time1))
    if Mask_Edge:
        sgbm_disparity[sobel_mask] = 0
    sgbm_disparity[sgbm_disparity < 0] = 0
    sgbm_disparity = sgbm_disparity / 3040 * 192.
    # plt.imsave("SGM1.png", sgbm_disparity, cmap='jet')

    # 将视差点进行过滤  并  转为三维点云
    f_u = 1.003556e+3
    baseline = 0.54
    # f_u = 1.057288e+3
    # baseline = 0.12

    mask = sgbm_disparity > 0
    depth_gt = f_u * baseline / (sgbm_disparity + 1. - mask)
    rows, cols = depth_gt.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth_gt])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud_gt = project_image_to_rect(points)

    # 将点云网格化
    min_mask = cloud_gt >= [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0]  # 2维的
    max_mask = cloud_gt <= [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size,
                            voxel_range[1] * voxel_size]
    if not eva:
        print("最小取值范围:", [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0], "\t", "最大取值范围:",
              [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size, voxel_range[1] * voxel_size])
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask  # 成为一个一维的
    filtered_cloud = cloud_gt[filter_mask]  # 全部过滤，速度更快  将体素网格以外的点进行过滤 过滤后剩余5966个点
    if not eva:
        print("共有{}个点\t其中有{}个点有效且在范围内\t雷达点云有效率为{}%。".format(points.shape[0], filtered_cloud.shape[0],
                                                            round((filtered_cloud.shape[0] / points.shape[0]) * 100,
                                                                  2)))

    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
    # vox_grid = np.zeros((int(32 / voxel_size), int(int(4 + H_mask) / voxel_size), int(24 / voxel_size)))  # 空的栅格地图
    vox_grid = np.zeros((int(voxel_range[0]), int(voxel_range[2]), int(voxel_range[1])))  # 空的栅格地图

    offsets = np.array([int(voxel_range[0] / 2), int(voxel_range[2] / 2), 0])  # 偏移量  与图片对齐，方便后续与图片的联合可视化
    xyz_offset_q = xyz_q + offsets  # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1  # 将空栅格通过索引赋值1 大小(256, 256, 32)
    if not eva:
        print("当前帧：深度图==》》Voxel成功！！！")
        print("预测体素网格完成，总耗时_{}！".format(time.time() - time1))

    # 下面的内容为可视化做准备
    return cloud_gt, vox_grid


def Depth_MobileStereo(Vis=True,
                       left_Path=r"F:\桌面文件\障碍物检测\障碍物检测程序\demo\left1.jpg",
                       right_Path=r"F:\桌面文件\障碍物检测\障碍物检测程序\demo\right1.jpg",
                       Save_index="MobileStereoNet", Mask_Edge=True, voxel_size=0.4, voxel_range=[64, 64, 16],
                       eva=False):
    from MobileStereoNet.MSNet3D import MSNet3D
    model = MSNet3D(192)  # 立体匹配
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    #state_dict = torch.load(r"MobileStereoNet/MSNet3D_SF_DS.ckpt")
    state_dict = torch.load(r"MobileStereoNet/MSNet3D_SF.ckpt")
    model.load_state_dict(state_dict['model'])
    if not eva:
        print("INFO  ==>>  MobileStereoNet Loading Success!")

        # 读取图片
        print("读取图片！")
    left_img = load_image(left_Path)
    right_img = load_image(right_Path)

    # 处理图片
    if not eva:
        print("处理图片！")
    w, h = left_img.size
    crop_w, crop_h = 880, 400
    left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
    if Mask_Edge:
        sobel_mask = Sobel_Mask(cv2.cvtColor(np.asarray(left_img), cv2.COLOR_RGB2GRAY), th=10)
    if Vis:
        # 这部分仅用于可视化
        points_rgb = np.transpose(np.asarray(left_img)[:, :, :3], (2, 0, 1)).reshape((3, -1)).T
    right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
    processed = get_transform()
    left_img = torch.unsqueeze(processed(left_img), 0)
    right_img = torch.unsqueeze(processed(right_img), 0)
    time1 = time.time()

    padder = InputPadder(left_img.shape, divis_by=64)
    left_img, right_img = padder.pad(left_img, right_img)

    with torch.no_grad():
        disparity = model(left_img.cuda(), right_img.cuda())[-1]
        disparity = padder.unpad(disparity.float()).squeeze()
        # disparity = torch.squeeze(disparity_list[-1])
        if Mask_Edge:
            disparity[sobel_mask] = 0
        disp_est_np = disparity.data.cpu().numpy()
        disp_est = np.array(disp_est_np, dtype=np.float32)
        disp_est[disp_est < 0] = 0
    if not eva:
        print("预测视差完成，耗时_{}！".format(time.time() - time1))

    # 将视差点进行过滤  并  转为三维点云
    f_u = 1.003556e+3
    baseline = 0.54
    # f_u = 1.057288e+3
    # baseline = 0.12

    mask = disp_est > 0
    depth_gt = f_u * baseline / (disp_est + 1. - mask)
    rows, cols = depth_gt.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth_gt])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud_gt = project_image_to_rect(points)

    # 将点云网格化
    min_mask = cloud_gt >= [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0]  # 2维的
    max_mask = cloud_gt <= [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size,
                            voxel_range[1] * voxel_size]
    if not eva:
        print("最小取值范围:", [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0], "\t", "最大取值范围:",
              [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size, voxel_range[1] * voxel_size])
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask  # 成为一个一维的
    filtered_cloud = cloud_gt[filter_mask]  # 全部过滤，速度更快  将体素网格以外的点进行过滤 过滤后剩余5966个点
    if not eva:
        print("共有{}个点\t其中有{}个点有效且在范围内\t雷达点云有效率为{}%。".format(points.shape[0], filtered_cloud.shape[0],
                                                            round((filtered_cloud.shape[0] / points.shape[0]) * 100,
                                                                  2)))

    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
    # vox_grid = np.zeros((int(32 / voxel_size), int(int(4 + H_mask) / voxel_size), int(24 / voxel_size)))  # 空的栅格地图
    vox_grid = np.zeros((int(voxel_range[0]), int(voxel_range[2]), int(voxel_range[1])))  # 空的栅格地图

    offsets = np.array([int(voxel_range[0] / 2), int(voxel_range[2] / 2), 0])  # 偏移量  与图片对齐，方便后续与图片的联合可视化
    xyz_offset_q = xyz_q + offsets  # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1  # 将空栅格通过索引赋值1 大小(256, 256, 32)
    if not eva:
        print("当前帧：深度图==》》Voxel成功！！！")
        print("预测体素网格完成，总耗时_{}！".format(time.time() - time1))

    return cloud_gt, vox_grid


def Depth_MobileStereo2D(Vis=True,
                         left_Path=r"F:\桌面文件\障碍物检测\障碍物检测程序\demo\left1.jpg",
                         right_Path=r"F:\桌面文件\障碍物检测\障碍物检测程序\demo\right1.jpg",
                         Save_index="MobileStereoNet", Mask_Edge=True, voxel_size=0.4, voxel_range=[64, 64, 16],
                         eva=False):
    from MobileStereoNet.MSNet2D import MSNet2D
    model = MSNet2D(192)  # 立体匹配
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    #state_dict = torch.load(r"MobileStereoNet/MSNet2D_SF_DS.ckpt")
    state_dict = torch.load(r"MobileStereoNet/MSNet2D_SF.ckpt")
    model.load_state_dict(state_dict['model'])
    if not eva:
        print("INFO  ==>>  MobileStereoNet2D Loading Success!")

        # 读取图片
        print("读取图片！")
    left_img = load_image(left_Path)
    right_img = load_image(right_Path)

    # 处理图片
    if not eva:
        print("处理图片！")
    w, h = left_img.size
    crop_w, crop_h = 880, 400
    left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
    if Mask_Edge:
        sobel_mask = Sobel_Mask(cv2.cvtColor(np.asarray(left_img), cv2.COLOR_RGB2GRAY), th=10)
    if Vis:
        # 这部分仅用于可视化
        points_rgb = np.transpose(np.asarray(left_img)[:, :, :3], (2, 0, 1)).reshape((3, -1)).T
    right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
    processed = get_transform()
    left_img = torch.unsqueeze(processed(left_img), 0)
    right_img = torch.unsqueeze(processed(right_img), 0)
    time1 = time.time()

    padder = InputPadder(left_img.shape, divis_by=64)
    left_img, right_img = padder.pad(left_img, right_img)

    with torch.no_grad():
        disparity = model(left_img.cuda(), right_img.cuda())[-1]
        disparity = padder.unpad(disparity.float()).squeeze()
        # disparity = torch.squeeze(disparity_list[-1])
        if Mask_Edge:
            disparity[sobel_mask] = 0
        disp_est_np = disparity.data.cpu().numpy()
        disp_est = np.array(disp_est_np, dtype=np.float32)
        disp_est[disp_est < 0] = 0
    if not eva:
        print("预测视差完成，耗时_{}！".format(time.time() - time1))

    # 将视差点进行过滤  并  转为三维点云
    f_u = 1.003556e+3
    baseline = 0.54
    # f_u = 1.057288e+3
    # baseline = 0.12

    mask = disp_est > 0
    depth_gt = f_u * baseline / (disp_est + 1. - mask)
    rows, cols = depth_gt.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth_gt])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]
    cloud_gt = project_image_to_rect(points)

    # 将点云网格化
    min_mask = cloud_gt >= [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0]  # 2维的
    max_mask = cloud_gt <= [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size,
                            voxel_range[1] * voxel_size]
    if not eva:
        print("最小取值范围:", [-(voxel_range[0] / 2) * voxel_size, -(voxel_range[2] / 2) * voxel_size, 0.0], "\t", "最大取值范围:",
              [(voxel_range[0] / 2) * voxel_size, (voxel_range[2] / 2) * voxel_size, voxel_range[1] * voxel_size])
    min_mask = min_mask[:, 0] & min_mask[:, 1] & min_mask[:, 2]
    max_mask = max_mask[:, 0] & max_mask[:, 1] & max_mask[:, 2]
    filter_mask = min_mask & max_mask  # 成为一个一维的
    filtered_cloud = cloud_gt[filter_mask]  # 全部过滤，速度更快  将体素网格以外的点进行过滤 过滤后剩余5966个点
    if not eva:
        print("共有{}个点\t其中有{}个点有效且在范围内\t雷达点云有效率为{}%。".format(points.shape[0], filtered_cloud.shape[0],
                                                            round((filtered_cloud.shape[0] / points.shape[0]) * 100,
                                                                  2)))

    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)  # 将点云转为栅格的整数
    # vox_grid = np.zeros((int(32 / voxel_size), int(int(4 + H_mask) / voxel_size), int(24 / voxel_size)))  # 空的栅格地图
    vox_grid = np.zeros((int(voxel_range[0]), int(voxel_range[2]), int(voxel_range[1])))  # 空的栅格地图

    offsets = np.array([int(voxel_range[0] / 2), int(voxel_range[2] / 2), 0])  # 偏移量  与图片对齐，方便后续与图片的联合可视化
    xyz_offset_q = xyz_q + offsets  # 添加偏移量，全部转为以0为起始单位,否则索引位置会出现错误
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1  # 将空栅格通过索引赋值1 大小(256, 256, 32)
    if not eva:
        print("当前帧：深度图==》》Voxel成功！！！")
        print("预测体素网格完成，总耗时_{}！".format(time.time() - time1))

    return cloud_gt, vox_grid

