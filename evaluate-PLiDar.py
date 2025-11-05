import torch
import numpy as np
from Depth_Baseline import Depth_GT, Depth_RAFT, Depth_TTF_P, Depth_SGM, Depth_BG, Depth_MobileStereo, \
    Depth_MobileStereo2D
import glob
from tqdm import tqdm
import time
import PIL.Image as Image
import torchvision.transforms as transforms


def Average(lst):
    return sum(lst) / len(lst)


def demo_val(model_name="SGBM", Mask_edge=True):
    iou_list_3 = []
    recall_list_3 = []
    precision_list_3 = []

    iou_list_2 = []
    recall_list_2 = []
    precision_list_2 = []
    time_list = []

    iou_list = []
    recall_list = []
    precision_list = []

    # left_path = r"F:\DataSets\DrivingStereo/valid_left/*/*.jpg"
    # right_path = r"F:\DataSets\DrivingStereo/valid_right/*/*.jpg"
    disp_path = r"F:\DataSets\DrivingStereo/GenerateValid_Disp/*/*.png"

    # left_images = sorted(glob.glob(left_path, recursive=True))
    # right_images = sorted(glob.glob(right_path, recursive=True))
    disp_images = sorted(glob.glob(disp_path, recursive=True))

    # print(len(left_images), len(right_images), len(disp_images))
    # assert len(left_images) == len(right_images) == len(disp_images)

    # for (imfile1, imfile2, dispfile) in tqdm(list(zip(left_images, right_images, disp_images))):
    for dispfile in tqdm(list(disp_images)):
        # print(dispfile)
        imfile1 = dispfile.replace("GenerateValid_Disp", "valid_left").replace("png", "jpg")
        imfile2 = dispfile.replace("GenerateValid_Disp", "valid_right").replace("png", "jpg")
        time1 = time.time()
        # 预测点云和体素
        if model_name == "MSN3D":
            cloud_pred, vox_pred = Depth_MobileStereo(Vis=False, voxel_size=0.2, Mask_Edge=Mask_edge,
                                                      voxel_range=[128, 128, 128], eva=True)
        if model_name == "BG":
            cloud_pred, vox_pred = Depth_BG(left_Path=imfile1, right_Path=imfile2, Vis=False, voxel_size=0.2,
                                            Mask_Edge=Mask_edge, voxel_range=[128, 128, 128])
        if model_name == "RAFT":
            cloud_pred, vox_pred = Depth_RAFT(left_Path=imfile1, right_Path=imfile2, Vis=False, voxel_size=0.2,
                                              voxel_range=[128, 128, 128], eva=True, Mask_edge=Mask_edge)
        elif model_name == "TTFP":
            cloud_pred, vox_pred = Depth_TTF_P(left_Path=imfile1, right_Path=imfile2, Vis=False, voxel_size=0.2,
                                               Mask_Edge=Mask_edge, voxel_range=[128, 128, 128], eva=True)
        # cloud_pred, vox_pred = Depth_TTF_P_Init(Vis=False, voxel_size=0.2, Mask_Edge=False,voxel_range=[64,64,32])

        elif model_name == "SGBM":
            cloud_pred, vox_pred = Depth_SGM(Mask_Edge=Mask_edge, left_Path=imfile1, right_Path=imfile2, Vis=False,
                                             voxel_size=0.2, voxel_range=[128, 128, 128], eva=True)
        elif model_name == "MSN2D":
            cloud_pred, vox_pred = Depth_MobileStereo2D(Mask_Edge=Mask_edge, left_Path=imfile1, right_Path=imfile2,
                                                        Vis=False,
                                                        voxel_size=0.2, voxel_range=[128, 128, 128], eva=True)
        # cloud_np_gt, vox_grid_gt = Depth_RAFT(Vis=False,voxel_size=0.4)

        time_consume = time.time() - time1
        time_list.append(time_consume)

        cloud_np_gt, vox_grid_gt = Depth_GT(left_Path=imfile1, disparity_Path=dispfile, Vis=False, voxel_size=0.2,
                                            voxel_range=[128, 128, 128], eav=True)

        intersect = vox_pred * vox_grid_gt  # 逻辑并集
        union = vox_pred + vox_grid_gt - intersect  # 逻辑交集

        IoU = (intersect.sum() + 1.) / (union.sum() + 1.)  # 交并比
        Recall = (intersect.sum() + 1.) / (vox_grid_gt.sum() + 1.)  # 找的全
        Precision = (intersect.sum() + 1.) / (vox_pred.sum() + 1.)  # 找的对

        # IoU = ((intersect.sum() + 1.0) / (union.sum() - intersect.sum() + 1.0))  # 交并比

        # cd = chamfer_distance(torch.Tensor(np.expand_dims(cloud_pred, 0)),
        # torch.Tensor(np.expand_dims(cloud_np_gt, 0)))  # 求两段点云的倒角距离

        iou_list.append(IoU)  # 添加到列表中
        recall_list.append(Recall)
        precision_list.append(Precision)

        # cd_list.append(cd)
        # print(f"CD is {Average(cd_list)}, IoU is {Average(iou_list)}")

        # _, vox_pred_2 = Depth_SGM(left_Path=imfile1, right_Path=imfile2, Vis=False, voxel_size=0.4,
        #                          voxel_range=[64, 64, 64], eva=True)
        # cloud_np_gt, vox_grid_gt = Depth_RAFT(Vis=False,voxel_size=0.4)
        # _, vox_grid_gt_2 = Depth_GT(left_Path=imfile1, disparity_Path=dispfile, Vis=False, voxel_size=0.4,
        #                            voxel_range=[64, 64, 64], eav=True)

        vox_pred_3 = vox_pred[25:121, 25:121, 25:121]
        vox_grid_gt_3 = vox_grid_gt[25:121, 25:121, 25:121]

        intersect_3 = vox_pred_3 * vox_grid_gt_3  # 逻辑并集
        union_3 = vox_pred_3 + vox_grid_gt_3 - intersect_3  # 逻辑交集

        IoU_3 = (intersect_3.sum() + 1.) / (union_3.sum() + 1.)  # 交并比
        Recall_3 = (intersect_3.sum() + 1.) / (vox_grid_gt_3.sum() + 1.)  # 找的全
        Precision_3 = (intersect_3.sum() + 1.) / (vox_pred_3.sum() + 1.)  # 找的对

        iou_list_3.append(IoU_3)  # 添加到列表中
        recall_list_3.append(Recall_3)
        precision_list_3.append(Precision_3)

        vox_pred_2 = vox_pred[31:95, 31:95, 31:95]
        vox_grid_gt_2 = vox_grid_gt[31:95, 31:95, 31:95]

        intersect_2 = vox_pred_2 * vox_grid_gt_2  # 逻辑并集
        union_2 = vox_pred_2 + vox_grid_gt_2 - intersect_2  # 逻辑交集

        IoU_2 = (intersect_2.sum() + 1.) / (union_2.sum() + 1.)  # 交并比
        Recall_2 = (intersect_2.sum() + 1.) / (vox_grid_gt_2.sum() + 1.)  # 找的全
        Precision_2 = (intersect_2.sum() + 1.) / (vox_pred_2.sum() + 1.)  # 找的对

        iou_list_2.append(IoU_2)  # 添加到列表中
        recall_list_2.append(Recall_2)
        precision_list_2.append(Precision_2)
    return time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3
    # print(f" Mean Inference Time is {Average(time_list)}")
    # print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
    #       f";  Precision is {Average(precision_list_2)}")
    # print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
    #       f";  Precision is {Average(precision_list_3)}")
    # print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
    #       f";  Precision is {Average(precision_list)}")


def load_image(filename):
    return Image.open(filename).convert('RGB')


def load_disp(filename):
    data = Image.open(filename)
    data = np.array(data, dtype=np.float32) / 256.
    return data


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])


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


import contextlib
import io


@contextlib.contextmanager
def suppress_stdout():
    with io.StringIO() as fake_stdout:
        with contextlib.redirect_stdout(fake_stdout):
            yield


print("SGBM, Mask_egde:False")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="SGBM", Mask_edge=False)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("SGBM, Mask_egde:True")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="SGBM", Mask_edge=True)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("BG, Mask_egde:False")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="BG", Mask_edge=False)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("BG, Mask_egde:True")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="BG", Mask_edge=True)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("RAFT, Mask_egde:False")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="RAFT", Mask_edge=False)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("RAFT, Mask_egde:True")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="RAFT", Mask_edge=True)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("TTFP, Mask_egde:False")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="TTFP", Mask_edge=False)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("TTFP, Mask_egde:True")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="TTFP", Mask_edge=True)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("MSN3D, Mask_egde:False, Fine=True")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="MSN3D", Mask_edge=False)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("MSN3D, Mask_egde:True, Fine=True")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="MSN3D", Mask_edge=True)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("MSN2D, Mask_egde:False, Fine=True")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="MSN2D", Mask_edge=False)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")

print("*" * 150)

print("MSN2D, Mask_egde:True, Fine=True")
with suppress_stdout():
    time_list, iou_list, recall_list, precision_list, iou_list_2, recall_list_2, precision_list_2, iou_list_3, recall_list_3, precision_list_3 = demo_val(
        model_name="MSN2D", Mask_edge=True)
print(f" Mean Inference Time is {Average(time_list)}")
print(f" 12.8 Extent:IoU is {Average(iou_list_2)}", f"; Recall is {Average(recall_list_2)}",
      f";  Precision is {Average(precision_list_2)}")
print(f" 19.2 Extent:IoU is {Average(iou_list_3)}", f"; Recall is {Average(recall_list_3)}",
      f";  Precision is {Average(precision_list_3)}")
print(f" 25.6 Extent:IoU is {Average(iou_list)}", f"; Recall is {Average(recall_list)}",
      f";  Precision is {Average(precision_list)}")
