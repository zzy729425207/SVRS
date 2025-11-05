import torch
import torch.nn as nn
import torch.nn.functional as F
from TTFStereo_performance.utils import bilinear_sampler, hourglass

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class build_cost_pyramid:
    def __init__(self, corr_cost, gwc_cost, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.gwc_cost_pyramid = []
        self.corr_cost_pyramid = []

        b1, c1, d1, h1, w1 = corr_cost.shape
        b, c, d, h, w = gwc_cost.shape
        assert c1 == 1 and b == b1 and d == d1 and h1 == h1 and w1 == w
        gwc_cost = gwc_cost.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, 1, d)
        corr_cost = corr_cost.permute(0, 3, 4, 1, 2).reshape(b * h * w, c1, 1, d)
        self.gwc_cost_pyramid.append(gwc_cost)
        self.corr_cost_pyramid.append(corr_cost)
        for i in range(self.num_levels - 1):
            gwc_cost = F.avg_pool2d(gwc_cost, [1, 2], stride=[1, 2])
            self.gwc_cost_pyramid.append(gwc_cost)
        for i in range(self.num_levels - 1):
            corr_cost = F.avg_pool2d(corr_cost, [1, 2], stride=[1, 2])
            self.corr_cost_pyramid.append(corr_cost)

    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.gwc_cost_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(1, 1, 2 * r + 1, 1).to(disp.device)
            x0 = dx + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0, y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.corr_cost_pyramid[i]
            init_x0 = coords.reshape(b * h * w, 1, 1, 1) / 2 ** i - disp.reshape(b * h * w, 1, 1, 1) / 2 ** i + dx
            init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
