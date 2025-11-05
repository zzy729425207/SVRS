import torch
import torch.nn as nn
import torch.nn.functional as F
from TTFStereo_performance.extractor import Mish

def activate(relu=True):
    if relu:
        print("ReLU loading...")
        return nn.ReLU(inplace=True)
    else:
        return Mish()


class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1, use_relu=True):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = activate(use_relu)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):  # 时候可以进行改进？
    def __init__(self, hidden_dim, input_dim, kernel_size=3):  # 128，128，3
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)  # 256，128
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)  # 256，128
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)  # 256，128

    def forward(self, h, cz, cr, cq, *x_list):  # h是上一个时刻的状态  net[1], *(inp[1]), pool2x(net[0])
        # 输入拆开   net[1]    inp[1][0]     inp[1][1]     inp[1][2]      pool2x(net[0])

        # print("cz",cz.shape)
        # print("cr",cr.shape)
        # print("cq",cq.shape)
        # print(len(x_list))
        x = torch.cat(x_list, dim=1)  # 当08时，会有motion_features, interp(net[1], net[0])两个作为元组再cat起来
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)  # 并不是纯正的GRU，而是RGU的变体，在里面加入了inp的固定参数
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius, num_groups):
        self.num_class = num_groups
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        super(BasicMotionEncoder, self).__init__()

        cor_planes = self.corr_levels * (self.num_class + 1) * (2 * self.corr_radius + 1)

        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 1, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))  # 处理corr 36-》64
        cor = F.relu(self.convc2(cor))  # 处理corr  64-》64
        flo = F.relu(self.convf1(flow))  # 处理flow 1-》64
        flo = F.relu(self.convf2(flo))  # 处理flow 64-》64

        cor_flo = torch.cat([cor, flo], dim=1)  # 叠加起来
        out = F.relu(self.conv(cor_flo))  # 处理叠加 128-》126
        return torch.cat([out, flow], dim=1)  # 把out于flow叠加起来


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)  # 双星号的是字典参数，这里意思是mode=bilinear,align_corner=True,下面为解释

    # https://blog.csdn.net/weixin_46457812/article/details/113877454?ops_request_misc=&request_id=&biz_id=102&utm_term=python%E5%8F%82%E6%95%B0%E5%89%8D%E9%9D%A2%E5%B8%A6%E6%98%9F%E5%8F%B7&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-113877454.142^v42^pc_rank_34_2,185^v2^control&spm=1018.2226.3001.4187


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, hidden_dims, n_gru_layers, n_downsample, use_relu=True, corr_level=1, corr_radius=4,
                 num_groups=8):
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_level, corr_radius, num_groups)
        encoder_output_dim = 128
        self.n_gru_layers = n_gru_layers

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (n_gru_layers > 1))  # 128,128+128
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (n_gru_layers == 3) + hidden_dims[2])  # 128,128
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1, use_relu=use_relu)
        factor = 2 ** n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            activate(use_relu),
            nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0))

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):
        # inp在GRU的forward中没有进行更新，只是作为参数参与前向传播
        # net 列表长度为2，每个为[1,128,90,40]或者[1,128,45,20] 来源于第一张图片特征的output08和output16
        # inp 列表维度为[2,3],每个为[1,128,90,40]或者[1,128,45,20],来源于第二张图片特征的output08和output16

        if iter32:  # False
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.n_gru_layers > 2:  # False
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))  # 16中也用到了08
            else:  # True  这个用于处理output16 net[1]
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))  # realtime从这里开始，对output08均值池化
                # net[1]
        if iter08:  # True
            motion_features = self.encoder(flow, corr)  # 对flow和corr进行编码
            if self.n_gru_layers > 1:  # 这个用于处理output08 net[0]
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))  # net是第一张图片的特征
                # 08中也用到了16，对net[1]按照net[0]的大小进行插值
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])  # 使用第一张图片进行预测delta_flow

        # scale mask to balence gradients 缩放mask来平衡梯度
        mask = .25 * self.mask(net[0])  # 为什么使用0.25？

        return net, mask, delta_flow  # net作为就是更新的变量
