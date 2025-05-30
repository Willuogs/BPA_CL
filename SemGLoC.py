import math
import pdb

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
import clip
from Text_Prompt01 import *
from tools import *
from einops import rearrange, repeat
from model.lib import ST_RenovateNet


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y
    


class Model_lst_4part(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=64, num_person=2, graph=None, graph_args=dict(), in_channels=3,base_channel=64,
                 drop_out=0, adaptive=True, head=['ViT-B/32'] ,cl_mode=None, multi_cl_weights=[1, 1, 1, 1], cl_version='V0', pred_threshold=0, use_p_map=True,
                 k=0):
        super(Model_lst_4part, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        self.A_vector = self.get_A(graph, k).float()


        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame

        self.num_person = num_person
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.adaptive = adaptive
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        if self.cl_mode is not None:
            self.build_cl_blocks()
        
        self.part_attn = nn.Sequential(
            nn.Linear(4 * 512, 128),  # 将拼接后的分区特征映射到中间维度
            nn.ReLU(),  # 引入非线性
            nn.Linear(128, 4),  # 输出每个分区的原始权重
            nn.Softmax(dim=-1)  # 对权重进行归一化
        )

        
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        
        


        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(self.base_channel, self.num_frame, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_mid = ST_RenovateNet(self.base_channel * 2, self.num_frame // 2, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_high = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_fin = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")
          
        
    def get_ST_Multi_Level_cl_output(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                  cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return logits, cl_loss

    def forward(self, x, label,**kwargs):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3,20]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,21,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,2,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        part_feats = torch.cat([head_feature, hand_feature, hip_feature, foot_feature], dim=1)
        part_weights = self.part_attn(part_feats)

        # 应用动态权重
        head_feature = head_feature * part_weights[:, 0].unsqueeze(1)
        hand_feature = hand_feature * part_weights[:, 1].unsqueeze(1)
        hip_feature = hip_feature * part_weights[:, 2].unsqueeze(1)
        foot_feature = foot_feature * part_weights[:, 3].unsqueeze(1)

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        logits, cl_loss = self.get_ST_Multi_Level_cl_output(x, feat_low, feat_mid, feat_high, feat_fin, label)
        
        return logits, cl_loss, feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class Model_lst_4part_bone(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=64, num_person=2, graph=None, graph_args=dict(), in_channels=3,base_channel=64,
                 drop_out=0, adaptive=True, head=['ViT-B/32'] ,cl_mode=None, multi_cl_weights=[1, 1, 1, 1], cl_version='V0', pred_threshold=0, use_p_map=True,
                 k=1):
        super(Model_lst_4part_bone, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        self.A_vector = self.get_A(graph, k).float()


        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame

        self.num_person = num_person
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.adaptive = adaptive
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        if self.cl_mode is not None:
            self.build_cl_blocks()
        
        self.part_attn = nn.Sequential(
            nn.Linear(4 * 512, 128),  # 将拼接后的分区特征映射到中间维度
            nn.ReLU(),  # 引入非线性
            nn.Linear(128, 4),  # 输出每个分区的原始权重
            nn.Softmax(dim=-1)  # 对权重进行归一化
        )

        
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        
        


        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(self.base_channel, self.num_frame, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_mid = ST_RenovateNet(self.base_channel * 2, self.num_frame // 2, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_high = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_fin = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")
          
        
    def get_ST_Multi_Level_cl_output(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                  cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return logits, cl_loss

    def forward(self, x, label,**kwargs):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,20,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        part_feats = torch.cat([head_feature, hand_feature, hip_feature, foot_feature], dim=1)
        part_weights = self.part_attn(part_feats)

        # 应用动态权重
        head_feature = head_feature * part_weights[:, 0].unsqueeze(1)
        hand_feature = hand_feature * part_weights[:, 1].unsqueeze(1)
        hip_feature = hip_feature * part_weights[:, 2].unsqueeze(1)
        foot_feature = foot_feature * part_weights[:, 3].unsqueeze(1)

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        logits, cl_loss = self.get_ST_Multi_Level_cl_output(x, feat_low, feat_mid, feat_high, feat_fin, label)
        
        return logits, cl_loss, feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class Model_lst_4part_ucla(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=52, num_person=2, graph=None, graph_args=dict(), in_channels=3,base_channel=64,
                 drop_out=0, adaptive=True, head=['ViT-B/32'] ,cl_mode=None, multi_cl_weights=[1, 1, 1, 1], cl_version='V0', pred_threshold=0, use_p_map=True,
                 k=0):
        super(Model_lst_4part_ucla, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        self.A_vector = self.get_A(graph, k).float()


        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame

        self.num_person = num_person
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.adaptive = adaptive
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        if self.cl_mode is not None:
            self.build_cl_blocks()
        
        self.part_attn = nn.Sequential(
            nn.Linear(4 * 512, 128),  # 将拼接后的分区特征映射到中间维度
            nn.ReLU(),  # 引入非线性
            nn.Linear(128, 4),  # 输出每个分区的原始权重
            nn.Softmax(dim=-1)  # 对权重进行归一化
        )

        
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        
        


        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(self.base_channel, self.num_frame, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_mid = ST_RenovateNet(self.base_channel * 2, self.num_frame // 2, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_high = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_fin = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")
          
        
    def get_ST_Multi_Level_cl_output(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                  cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return logits, cl_loss

    def forward(self, x, label,**kwargs):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3]).long()
        hand_list = torch.Tensor([10,11,6,7,8,9,4,5]).long()
        foot_list = torch.Tensor([16,17,18,19,12,13,14,15]).long()
        hip_list = torch.Tensor([0,1,12,16]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        part_feats = torch.cat([head_feature, hand_feature, hip_feature, foot_feature], dim=1)
        part_weights = self.part_attn(part_feats)

        # 应用动态权重
        head_feature = head_feature * part_weights[:, 0].unsqueeze(1)
        hand_feature = hand_feature * part_weights[:, 1].unsqueeze(1)
        hip_feature = hip_feature * part_weights[:, 2].unsqueeze(1)
        foot_feature = foot_feature * part_weights[:, 3].unsqueeze(1)

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        logits, cl_loss = self.get_ST_Multi_Level_cl_output(x, feat_low, feat_mid, feat_high, feat_fin, label)
        
        return logits, cl_loss, feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class Model_lst_4part_bone_ucla(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_frame=52, num_person=2, graph=None, graph_args=dict(), in_channels=3,base_channel=64,
                 drop_out=0, adaptive=True, head=['ViT-B/32'] ,cl_mode=None, multi_cl_weights=[1, 1, 1, 1], cl_version='V0', pred_threshold=0, use_p_map=True,
                 k=1):
        super(Model_lst_4part_bone_ucla, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25
        self.A_vector = self.get_A(graph, k).float()


        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame

        self.num_person = num_person
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.adaptive = adaptive
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        if self.cl_mode is not None:
            self.build_cl_blocks()
        
        self.part_attn = nn.Sequential(
            nn.Linear(4 * 512, 128),  # 将拼接后的分区特征映射到中间维度
            nn.ReLU(),  # 引入非线性
            nn.Linear(128, 4),  # 输出每个分区的原始权重
            nn.Softmax(dim=-1)  # 对权重进行归一化
        )

        
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        
        


        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList()

        for i in range(4):
            self.part_list.append(nn.Linear(256,512))

        self.head = head
        if 'ViT-B/32' in self.head:
            self.linear_head['ViT-B/32'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/32'])
        
        if 'ViT-B/16' in self.head:
            self.linear_head['ViT-B/16'] = nn.Linear(256,512)
            conv_init(self.linear_head['ViT-B/16'])
        if 'ViT-L/14' in self.head:
            self.linear_head['ViT-L/14'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14'])
        if 'ViT-L/14@336px' in self.head:
            self.linear_head['ViT-L/14@336px'] = nn.Linear(256,768)
            conv_init(self.linear_head['ViT-L/14@336px'])
        
        if 'RN50x64' in self.head:
            self.linear_head['RN50x64'] = nn.Linear(256,1024)
            conv_init(self.linear_head['RN50x64'])

        if 'RN50x16' in self.head:
            self.linear_head['RN50x16'] = nn.Linear(256,768)
            conv_init(self.linear_head['RN50x16'])

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(self.base_channel, self.num_frame, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_mid = ST_RenovateNet(self.base_channel * 2, self.num_frame // 2, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_high = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_fin = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")
          
        
    def get_ST_Multi_Level_cl_output(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                  cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return logits, cl_loss

    def forward(self, x, label,**kwargs):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2]).long()
        hand_list = torch.Tensor([7,8,9,10,3,4,5,6]).long()
        foot_list = torch.Tensor([11,12,13,14,15,16,17,18]).long()
        hip_list = torch.Tensor([0,1,11,15]).long()
        head_feature = self.part_list[0](feature[:,:,:,:,head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[2](feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[3](feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1))


        part_feats = torch.cat([head_feature, hand_feature, hip_feature, foot_feature], dim=1)
        part_weights = self.part_attn(part_feats)

        # 应用动态权重
        head_feature = head_feature * part_weights[:, 0].unsqueeze(1)
        hand_feature = hand_feature * part_weights[:, 1].unsqueeze(1)
        hip_feature = hip_feature * part_weights[:, 2].unsqueeze(1)
        foot_feature = foot_feature * part_weights[:, 3].unsqueeze(1)

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)
        
        x = self.drop_out(x)

        logits, cl_loss = self.get_ST_Multi_Level_cl_output(x, feat_low, feat_mid, feat_high, feat_fin, label)
        
        return logits, cl_loss, feature_dict, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]