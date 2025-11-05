import torch
import torch.nn as nn
import torch.nn.functional as F
from TTFStereo_performance.update import BasicMultiUpdateBlock
from TTFStereo_performance.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock, activate
from TTFStereo_performance.utils import coords_grid, upflow8, build_gwc_volume, convbn_3d, hourglass, \
    disparity_regression
from TTFStereo_performance.corr import build_cost_pyramid

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class TTFStereo_performance(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dims = [128] * 3
        self.context_dims = [128] * 3
        self.n_downsample = 2  # 上采样的个数，就是三，不要动
        self.n_gru_layers = 2  # 1 GRU的层数
        self.iters = 8  # 8 训练循环次数 5是最具有性价比的选择
        self.corr_radius = 4
        self.use_relu = True
        self.cost_pyramid_level = 2
        self.max_disp = 192
        self.num_groups = 16  # 16  分类

        self.cnet = MultiBasicEncoder(output_dim=[self.hidden_dims, self.context_dims], norm_fn="domain",downsample=self.n_downsample, use_relu=self.use_relu)
        self.update_block = BasicMultiUpdateBlock(hidden_dims=self.hidden_dims, n_gru_layers=self.n_gru_layers,
                                                  n_downsample=self.n_downsample, use_relu=self.use_relu,
                                                  corr_level=self.cost_pyramid_level, corr_radius=self.corr_radius,
                                                  num_groups=self.num_groups)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(self.context_dims[i], self.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.n_gru_layers)])

        self.conv2 = nn.Sequential(
            ResidualBlock(128, 128, 'domain', stride=1, use_relu=self.use_relu),
            nn.Conv2d(128, 256, 3, padding=1))
        self.dres0 = nn.Sequential(convbn_3d(self.num_groups, self.num_groups, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(self.num_groups, self.num_groups, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(self.num_groups, self.num_groups, 3, 1, 1),
                                   activate(relu=self.use_relu),
                                   convbn_3d(self.num_groups, self.num_groups, 3, 1, 1))
        self.cost_agg = hourglass(self.num_groups)
        self.classifier = nn.Sequential(nn.Conv3d(self.num_groups+1, self.num_groups+1, 3, 1, 1, bias=False),  # 用于初始化视差
                                                        activate(relu=self.use_relu),
                                                        nn.Conv3d(self.num_groups+1, 1, 3, 1, 1, bias=False))
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_flow(self, disp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * disp, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, test_mode=False):
        """ Estimate optical flow between pair of frames """
            # run the context network
        *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.n_gru_layers)
        fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)

        net_list = [torch.tanh(x[0]) for x in cnet_list]  # 第一张图片的特征使用tanh进行激活
        inp_list = [torch.relu(x[1]) for x in cnet_list]  # 第二章图片的特征使用relu进行激活,三个，output08,output16

        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                    zip(inp_list, self.context_zqr_convs)]  # 把128拓展成384，再分割成3个

        fmap1, fmap2 = fmap1.float(), fmap2.float()
        gwc_volume = build_gwc_volume(fmap1, fmap2, maxdisp=self.max_disp // 2 ** self.n_downsample,
                                      num_groups=self.num_groups)
        gwc_volume_cost = self.dres0(gwc_volume)
        gwc_volume_cost = self.dres1(gwc_volume_cost) + gwc_volume

        corr_volume_cost = gwc_volume_cost.mean(dim=1, keepdims=True)  # 3D代价体积  [1,1,32,40,80]
        
        gwc_volume_cost = self.cost_agg(gwc_volume_cost)  # 聚合完成的代价体积  [1,8,32,40,80]

        init_disp0 = disparity_regression(F.softmax(self.classifier(torch.cat((gwc_volume_cost,corr_volume_cost),1)).squeeze(1), dim=1),
                                                          self.max_disp // 2 ** self.n_downsample)  # 初始化的视差

        # 建立体积金字塔，建立
        cost_bolck = build_cost_pyramid
        cost_fn = cost_bolck(corr_volume_cost, gwc_volume_cost, num_levels=self.cost_pyramid_level,
                             radius=self.corr_radius)
        b, c, h, w = fmap1.shape
        coords = torch.arange(w).float().to(fmap1.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp0
        disp_preds = []
        for itr in range(self.iters):  # 从下面开始是GRU的部分
            disp = disp.detach()
            cost_feat = cost_fn(disp, coords)  # index correlation volume,这里是拿出值来  [[1, 81, 40, 80]]
            # 最后一部分，多级更新器
            net_list, up_mask, delta_disp = self.update_block(net_list, inp_list, cost_feat, disp,
                                                          iter32=self.n_gru_layers == 3,
                                                          iter16=self.n_gru_layers >= 2)  # iters32 False\iters16 True
            disp = disp + delta_disp

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < self.iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                disp_up = upflow8(disp)
            else:
                disp_up = self.upsample_flow(disp, up_mask)  # upmask不是None，所以对光流和upmask进行上采样

            disp_preds.append(disp_up)

        if test_mode:
            # return coords1 - coords0, flow_up
            return [disp_up]
        return F.upsample(init_disp0 * (2 ** self.n_downsample), (image1.size()[2], image2.size()[3]), mode="bilinear"), disp_preds


class TTFStereo_performance_Init(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dims = [128] * 3
        self.context_dims = [128] * 3
        self.n_downsample = 2  # 上采样的个数，就是三，不要动
        self.n_gru_layers = 2  # 1 GRU的层数
        self.iters = 8  # 8 训练循环次数 5是最具有性价比的选择
        self.corr_radius = 4
        self.use_relu = True
        self.cost_pyramid_level = 2
        self.max_disp = 192
        self.num_groups = 16  # 16  分类

        self.cnet = MultiBasicEncoder(output_dim=[self.hidden_dims, self.context_dims], norm_fn="domain",
                                      downsample=self.n_downsample, use_relu=self.use_relu)
        self.update_block = BasicMultiUpdateBlock(hidden_dims=self.hidden_dims, n_gru_layers=self.n_gru_layers,
                                                  n_downsample=self.n_downsample, use_relu=self.use_relu,
                                                  corr_level=self.cost_pyramid_level, corr_radius=self.corr_radius,
                                                  num_groups=self.num_groups)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(self.context_dims[i], self.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.n_gru_layers)])

        self.conv2 = nn.Sequential(
            ResidualBlock(128, 128, 'domain', stride=1, use_relu=self.use_relu),
            nn.Conv2d(128, 256, 3, padding=1))
        self.dres0 = nn.Sequential(convbn_3d(self.num_groups, self.num_groups, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(self.num_groups, self.num_groups, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(self.num_groups, self.num_groups, 3, 1, 1),
                                   activate(relu=self.use_relu),
                                   convbn_3d(self.num_groups, self.num_groups, 3, 1, 1))
        self.cost_agg = hourglass(self.num_groups)
        self.classifier = nn.Sequential(nn.Conv3d(self.num_groups + 1, self.num_groups + 1, 3, 1, 1, bias=False),
                                        # 用于初始化视差
                                        activate(relu=self.use_relu),
                                        nn.Conv3d(self.num_groups + 1, 1, 3, 1, 1, bias=False))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_flow(self, disp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = disp.shape
        factor = 2 ** self.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * disp, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # run the context network
        *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.n_gru_layers)
        fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)

        fmap1, fmap2 = fmap1.float(), fmap2.float()
        gwc_volume = build_gwc_volume(fmap1, fmap2, maxdisp=self.max_disp // 2 ** self.n_downsample,
                                      num_groups=self.num_groups)
        gwc_volume_cost = self.dres0(gwc_volume)
        gwc_volume_cost = self.dres1(gwc_volume_cost) + gwc_volume

        corr_volume_cost = gwc_volume_cost.mean(dim=1, keepdims=True)  # 3D代价体积  [1,1,32,40,80]

        gwc_volume_cost = self.cost_agg(gwc_volume_cost)  # 聚合完成的代价体积  [1,8,32,40,80]

        init_disp0 = disparity_regression(
            F.softmax(self.classifier(torch.cat((gwc_volume_cost, corr_volume_cost), 1)).squeeze(1), dim=1),
            self.max_disp // 2 ** self.n_downsample)  # 初始化的视差

        return F.upsample(init_disp0 * (2 ** self.n_downsample), (image1.size()[2], image2.size()[3]))