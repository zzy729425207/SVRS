import torch
import torch.nn as nn
from dataset import VoxelTestDSDataset
from torch.utils.data import DataLoader
from model import Voxel2D_Norm, Voxel2D_Hie, Voxel2D_Hie_Mask, Voxel2D_Hie_Mask_Group, Voxel2D_Hie_Mask_Occ, \
    Voxel2D_Hie_Mask_Occ0506, Voxel2D_Hie_Mask_Group0509, Voxel2D_Hie_Mask_Group_Domain,Voxel2D_Hie_Mask_GroupV3
from utils import model_loss, tensor2float, tensor2numpy
from tqdm import tqdm
import numpy as np
import time

torch.backends.cudnn.benchmark = True

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def Average(lst):
    return round(sum(lst) / len(lst), 4)

print(torch.cuda.is_available())

model = Voxel2D_Hie_Mask_GroupV3(192, "voxel")
checkpoints_path ="/home/johndoe/data-1/selfstereovoxel/stereovoxel_train/checkpoints/Voxel2D_Hie_Mask_GroupV3/25_Voxel2D_Hie_Mask_GroupV3.pth"
data_path = "/home/johndoe/data-1/selfstereovoxel/dataset/"
test_list = "/home/johndoe/data-1/selfstereovoxel/dataset/Valid_DrivingStereo.txt"

batch_size = 1
loader_workers = 8
VOXEL_SIZE = 0.2
c_u = 4.556890e+2
c_v = 1.976634e+2
f_u = 1.003556e+3
f_v = 1.003556e+3
baseline = 0.54


def test_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, voxel_gt, voxel_cost_vol = sample['left'], sample['right'], sample['voxel_grid'], sample[
        'vox_cost_vol_disps']
    if torch.cuda.is_available():
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        for i in range(len(voxel_gt)):
            voxel_gt[i] = voxel_gt[i].cuda()

    voxel_ests = model(imgL, imgR, voxel_cost_vol)  #
    loss, iou = model_loss(voxel_ests, voxel_gt)

    scalar_outputs = {"loss": loss}
    if compute_metrics:
        scalar_outputs["IoU"] = iou

    return tensor2float(loss), tensor2float(scalar_outputs)


StereoDataset = VoxelTestDSDataset
test_dataset = StereoDataset(data_path, test_list, True, True, False)

# TrainImgLoader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=loader_workers, drop_last=True,pin_memory=True, persistent_workers=True, prefetch_factor=4)
TestImgLoader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=loader_workers, drop_last=False)

model = nn.DataParallel(model, device_ids=[0])

if checkpoints_path is not None:
    if checkpoints_path.endswith(".pth"):
        print("INFO:loading model form {}".format(checkpoints_path))
        checkpoint = torch.load(checkpoints_path)
        model.load_state_dict(checkpoint, strict=True)
        print("INFO:loading pth success")
    else:
        print("INFO:loading model form {}".format(checkpoints_path))
        state_dict = torch.load(checkpoints_path)
        model.load_state_dict(state_dict['model'])

if torch.cuda.is_available():
    model.cuda()

model.eval()

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

total_count = len(TestImgLoader) * batch_size

t = tqdm(TestImgLoader)

for batch_idx, sample in enumerate(t):
    left_img, right_img, disparity_batch, left_filename, vox_cost_vol_disps = sample['left'], sample['right'], \
                                                                              sample['disparity'], sample[
                                                                                  'left_filename'], sample[
                                                                                  'vox_cost_vol_disps']
    voxel_grids = sample["voxel_grid"]

    with torch.no_grad():
        time1 = time.time()
        vox_preds = model(left_img.cuda(), right_img.cuda(), vox_cost_vol_disps)[0][-1]
    time_consume = time.time() - time1
    vox_grid_gt = voxel_grids[-1].cpu().numpy()
    vox_pred = vox_preds.cpu().numpy()
    vox_pred[vox_pred < 0.5] = 0
    vox_pred[vox_pred >= 0.5] = 1

    intersect = vox_pred * vox_grid_gt  # 逻辑并集
    union = vox_pred + vox_grid_gt - intersect  # 逻辑交集
    
    
    IoU = (intersect.sum()) / (union.sum())  # 交并比
    Recall = (intersect.sum()) / (vox_grid_gt.sum())  # 找的全
    Precision = (intersect.sum()) / (vox_pred.sum())  # 找的对
    time_list.append(time_consume)
    
    if vox_pred.sum()!=0 and union.sum()!=0 and vox_grid_gt.sum()!=0: 
        iou_list.append(IoU)  # 添加到列表中
        recall_list.append(Recall)
        precision_list.append(Precision)

    vox_pred_2 = vox_pred[:, 25:121, 25:121, 25:121]  # 25.6 31-95  12.8 47-79
    vox_grid_gt_2 = vox_grid_gt[:, 25:121, 25:121, 25:121]

    intersect_2 = vox_pred_2 * vox_grid_gt_2  # 逻辑并集
    union_2 = vox_pred_2 + vox_grid_gt_2 - intersect_2  # 逻辑交集

    IoU_2 = (intersect_2.sum()) / (union_2.sum())  # 交并比
    Recall_2 = (intersect_2.sum()) / (vox_grid_gt_2.sum())  # 找的全
    Precision_2 = (intersect_2.sum()) / (vox_pred_2.sum())  # 找的对
    
    if vox_pred_2.sum()!=0 and union_2.sum()!=0 and vox_grid_gt_2.sum()!=0:
        iou_list_2.append(IoU_2)  # 添加到列表中
        recall_list_2.append(Recall_2)
        precision_list_2.append(Precision_2)
    
    # 25.6
    vox_pred_3 = vox_pred[:, 31:95, 31:95, 31:95]  # 25.6 31-95  12.8 47-79
    vox_grid_gt_3 = vox_grid_gt[:, 31:95, 31:95, 31:95]

    intersect_3 = vox_pred_3 * vox_grid_gt_3  # 逻辑并集
    union_3 = vox_pred_3 + vox_grid_gt_3 - intersect_3  # 逻辑交集

    IoU_3 = (intersect_3.sum()) / (union_3.sum())  # 交并比
    Recall_3 = (intersect_3.sum()) / (vox_grid_gt_3.sum())  # 找的全
    Precision_3 = (intersect_3.sum()) / (vox_pred_3.sum())  # 找的对
    
    if vox_pred_3.sum()!=0 and union_3.sum()!=0 and vox_grid_gt_3.sum()!=0:
        iou_list_3.append(IoU_3)  # 添加到列表中
        recall_list_3.append(Recall_3)
        precision_list_3.append(Precision_3)

    # cd = chamfer_distance(torch.Tensor(np.expand_dims(cloud_pred, 0)),
    #                      torch.Tensor(np.expand_dims(cloud_np_gt, 0)))[0]
    # cd_list.append(cd)

    # t.set_description(
    # f"CD is {Average(cd_list)}, IoU is {Average(iou_list)}, Invalid Sample {invalid_count} out of {total_count} @ {round(invalid_count / total_count * 100, 2)}%")
    t.set_description(
        f" Mean Inference Time is {Average(time_list)} ; 12.8 Extent:IoU is {Average(iou_list_3)} ; Recall is {Average(recall_list_3)} ; Precision is {Average(precision_list_3)} === 25.6 Extent:IoU is {Average(iou_list_2)} ; Recall is {Average(recall_list_2)} ; Precision is {Average(precision_list_2)} === 51.2 Extent:IoU is {Average(iou_list)} ; Recall is {Average(recall_list)} ; Precision is {Average(precision_list)} ")
    t.refresh()
print(f" Mean Inference Time is {Average(time_list)} ; 12.8 Extent:IoU is {Average(iou_list_3)} ; Recall is {Average(recall_list_3)} ; Precision is {Average(precision_list_3)} === 25.6 Extent:IoU is {Average(iou_list_2)} ; Recall is {Average(recall_list_2)} ; Precision is {Average(precision_list_2)} === 51.2 Extent:IoU is {Average(iou_list)} ; Recall is {Average(recall_list)} ; Precision is {Average(precision_list)} ")
