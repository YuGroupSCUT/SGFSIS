"""
Func: Replacing the GCM module with a naive structure
Author: my
Data: 2023/08/21
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from collections import OrderedDict
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet
import random

manual_seed = 321
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)

class Net(nn.Module):
    """ A base class provides a common weight initialisation scheme."""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x

####
class TFSamepaddingLayer(nn.Module):
    """To align with tf `same` padding.

    Putting this before any conv layer that need padding
    Assuming kernel has Height == Width for simplicity
    """

    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - (x.shape[2] % self.stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)
        # print(x.shape, padding)
        x = F.pad(x, padding, "constant", 0)
        # print(x.shape)
        return x


####
class DenseBlock(Net):
    """Dense Block as defined in:

    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger.
    "Densely connected convolutional networks." In Proceedings of the IEEE conference
    on computer vision and pattern recognition, pp. 4700-4708. 2017.

    Only performs `valid` convolution.

    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super(DenseBlock, self).__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        pad_vals = [v // 2 for v in unit_ksize]
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    nn.BatchNorm2d(unit_in_ch, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_in_ch, unit_ch[0], unit_ksize[0],
                        stride=1, padding=pad_vals[0], bias=False,
                    ),
                    nn.BatchNorm2d(unit_ch[0], eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_ch[0], unit_ch[1], unit_ksize[1],
                        stride=1, padding=pad_vals[1], bias=False,
                        groups=split,
                    ),
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            nn.BatchNorm2d(unit_in_ch, eps=1e-5),
            nn.ReLU(inplace=True)
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat


class UpSample2x(nn.Module):
    """Upsample input by a factor of 2.

    Assume input is of NCHW, port FixedUnpooling from TensorPack.
    """

    def __init__(self):
        super(UpSample2x, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


class ResNetExt(ResNet):
    def _forward_impl(self, x, freeze):
        # See note [TorchScript super()]
        if self.training:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.layer1(x)
                x2 = x = self.layer2(x)
                x3 = x = self.layer3(x)
                x4 = x = self.layer4(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor, freeze: bool = False) -> torch.Tensor:
        return self._forward_impl(x, freeze)

    @staticmethod
    def resnet50(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3])
        model.conv1 = nn.Conv2d(
            num_input_channels, 64, 7, stride=1, padding=3)
        if pretrained is not None and os.path.exists(pretrained):
            print(f"Loading: {pretrained}")
            pretrained = torch.load(pretrained)
            (
                missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)

        elif not os.path.exists(pretrained):
            assert os.path.exists(pretrained), \
                f"Pretrained path is not valid: {pretrained}"
        return model


class SGFSL(nn.Module):
    def __init__(self,
                 modal_num,
                 num_types=6,
                 freeze=False,
                 pretrained_backbone="/home/data1/my/Project/SGFSL/Init_model/resnet50-0676ba61.pth"):
                 # criterion=nn.CrossEntropyLoss(ignore_index=255)):
        super(SGFSL, self).__init__()
        self.freeze = freeze
        self.num_types = num_types
        self.modal_num = modal_num
        self.backbone = ResNetExt.resnet50(num_input_channels=3, pretrained=pretrained_backbone)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            pad = ksize // 2
            module_list = [
                nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(256, [1, ksize], [128, 32], 8, split=4),
                nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            ]
            u3 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(128, [1, ksize], [128, 32], 4, split=4),
                nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            ]
            u2 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
            ]
            u1 = nn.Sequential(*module_list)

            module_list = [
                nn.BatchNorm2d(64, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
            ]
            u0 = nn.Sequential(*module_list)

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
            )
            return decoder

        ksize = 3
        assert num_types != None
        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                    ("tp", create_decoder_branch(ksize=ksize, out_ch=num_types)),
                    ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ("conp", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ("cenp", create_decoder_branch(ksize=ksize, out_ch=1)),
                ]
            )
        )
        self.upsample2x = UpSample2x()

        # set the structure prototype (bg/contour/... e.g. points)
        # init memory bank with orthogonal reg

        sg_dim = 64
        # semantic structure guidance prototypes
        self.contour_proto_bank = nn.Parameter(torch.randn(1 + self.modal_num, sg_dim).cuda())
        self.semantic_proto_bank = nn.Parameter(torch.randn(1 + self.modal_num, sg_dim).cuda())
        # contour structure guidance prototypes, bg/fore are both defined
        self.centroid_proto_bank = nn.Parameter(torch.randn(1 + self.modal_num, sg_dim).cuda())

        self.semp_guided_conv = nn.Sequential(
             nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False),
             nn.LeakyReLU(inplace=True)
        )
        self.cenp_guided_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.conp_guided_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x=None,
                y_cls=None,
                y_fore=None,
                y_con=None,
                y_cen=None,
                gened_sem_proto=None,
                gened_cen_proto=None,
                gened_con_proto=None,
                meta_train_model=False,
                gen_proto=False,
                eval_model=False,
                eval_with_sg=False,
                novel_list=None,
                base_list=None):


        def generate_sem_structure_proto(x, y_fore):
            # batch_size, channel, h, w
            b, c, h, w = x.size()[:]

            tmp_y = F.interpolate(y_fore.float().unsqueeze(1), size=(h, w), mode='nearest')
            # semantic prototype
            tmp_mask = (tmp_y != 0).float()
            tmp_neg_mask = (tmp_y == 0).float()
            sup_sem_feat_list = []
            # if there is no foreground obj on input image
            if torch.sum(tmp_mask) == 0 or torch.sum(tmp_neg_mask) == 0:
                sup_sem_feat_list += [self.semantic_proto_bank.data.clone().unsqueeze(0)]
            else:
                # semantic mask average pooling on bs channel
                b = tmp_mask.shape[0] # batch size: fake num
                for i in range(b):
                    tmp_mask_bs = tmp_mask[i, :, :, :]
                    tmp_neg_mask_bs = tmp_neg_mask[i, :, :, :]
                    if torch.sum(tmp_mask_bs) != 0 and torch.sum(tmp_neg_mask_bs) != 0:
                        pos_proto = ((x[i, :, :, :] * tmp_mask_bs).sum(-1).sum(-1) /
                             (tmp_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        neg_proto = ((x[i, :, :, :] * tmp_neg_mask_bs).sum(-1).sum(-1) /
                             (tmp_neg_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        sup_sem_feat_list += [torch.cat([neg_proto, pos_proto], dim=0).unsqueeze(0)]

            return sup_sem_feat_list

        def generate_cen_structure_proto(x, y_cen):
            # batch_size, channel, h, w
            b, c, h, w = x.size()[:]

            # center prototype
            sup_cen_feat_list = []
            tmp_y_cen = F.interpolate(y_cen.float().unsqueeze(1), size=(h, w), mode='nearest')
            tmp_cen_mask = (tmp_y_cen != 0).float()
            tmp_neg_cen_mask = (tmp_y_cen == 0).float()
            if torch.sum(tmp_cen_mask) == 0 or torch.sum(tmp_neg_cen_mask) == 0:
                sup_cen_feat_list += [self.centroid_proto_bank.data.clone().unsqueeze(0)]
            else:
                # semantic mask average pooling on bs channel
                b = tmp_cen_mask.shape[0]  # batch size: fake num
                for i in range(b):
                    tmp_cen_mask_bs = tmp_cen_mask[i, :, :, :]
                    tmp_neg_cen_mask_bs = tmp_neg_cen_mask[i, :, :, :]
                    if torch.sum(tmp_cen_mask_bs) != 0 and torch.sum(tmp_neg_cen_mask_bs) != 0:
                        pos_proto = ((x[i, :, :, :] * tmp_cen_mask_bs).sum(-1).sum(-1) /
                                     (tmp_cen_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        neg_proto = ((x[i, :, :, :] * tmp_neg_cen_mask_bs).sum(-1).sum(-1) /
                                     (tmp_neg_cen_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        sup_cen_feat_list += [torch.cat([neg_proto, pos_proto], dim=0).unsqueeze(0)]

            return sup_cen_feat_list

        def generate_con_structure_proto(x, y_con):
            # batch_size, channel, h, w
            b, c, h, w = x.size()[:]
            # contour prototype
            sup_con_feat_list = []

            tmp_y_con = F.interpolate(y_con.float().unsqueeze(1), size=(h, w), mode='nearest')
            tmp_con_mask = (tmp_y_con != 0).float()
            tmp_neg_con_mask = (tmp_y_con == 0).float()

            if torch.sum(tmp_con_mask) == 0 or torch.sum(tmp_neg_con_mask) == 0:
                sup_con_feat_list += [self.contour_proto_bank.data.clone().unsqueeze(0)]
            else:
                # contour mask average pooling on bs channel
                for i in range(b):
                    tmp_con_mask_bs = tmp_con_mask[i, :, :, :]
                    tmp_neg_con_mask_bs = tmp_neg_con_mask[i, :, :, :]
                    if torch.sum(tmp_con_mask_bs) != 0 and torch.sum(tmp_neg_con_mask_bs) != 0:
                        pos_proto = ((x[i, :, :, :] * tmp_con_mask_bs).sum(-1).sum(-1) /
                                     (tmp_con_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        neg_proto = ((x[i, :, :, :] * tmp_neg_con_mask_bs).sum(-1).sum(-1) /
                                     (tmp_neg_con_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        sup_con_feat_list += [torch.cat([neg_proto, pos_proto], dim=0).unsqueeze(0)]

            return sup_con_feat_list


        if gen_proto:
            # proto generation
            # supp_x: [8, cls, s, c, h, w]
            # supp_y: [8, cls, s, h, w]

            assert novel_list != None and base_list != None
            x = x[0]
            y = y_cls[0]
            y_con = y_con[0]
            y_cen = y_cen[0]

            with torch.no_grad():

                tmp_x = x

                d0, d1, d2, d3 = self.backbone(tmp_x, self.freeze)
                d3 = self.conv_bot(d3)
                d = [d0, d1, d2, d3]
                u3 = self.upsample2x(d[-1]) + d[-2]

                # np sg feat map
                sg_u3 = self.decoder['np'][0](u3)
                sg_u2 = self.upsample2x(sg_u3) + d[-3]
                sg_u2 = self.decoder['np'][1](sg_u2)
                sg_u1 = self.upsample2x(sg_u2) + d[-4]
                tmp_np_sg_x = self.decoder['np'][2](sg_u1)
                tmp_sem_sg_x = self.semp_guided_conv(tmp_np_sg_x)

                # cen sg feat map
                sg_cen_u3 = self.decoder['cenp'][0](u3)
                sg_cen_u2 = self.upsample2x(sg_cen_u3) + d[-3]
                sg_cen_u2 = self.decoder['cenp'][1](sg_cen_u2)
                sg_cen_u1 = self.upsample2x(sg_cen_u2) + d[-4]
                tmp_cen_sg_x = self.decoder['cenp'][2](sg_cen_u1)
                tmp_cen_sg_x = self.cenp_guided_conv(tmp_cen_sg_x)

                # con sg feat map
                sg_con_u3 = self.decoder['conp'][0](u3)
                sg_con_u2 = self.upsample2x(sg_con_u3) + d[-3]
                sg_con_u2 = self.decoder['conp'][1](sg_con_u2)
                sg_con_u1 = self.upsample2x(sg_con_u2) + d[-4]
                tmp_con_sg_x = self.decoder['conp'][2](sg_con_u1)
                tmp_con_sg_x = self.conp_guided_conv(tmp_con_sg_x)

                # batch_size, channel, h, w
                b, c, h, w = tmp_sem_sg_x.size()[:]

                ######################### generate semantic mask average pooling on bs channel ######################
                tmp_y_fore = F.interpolate(y.float().unsqueeze(1), size=(h, w), mode='nearest')
                tmp_mask = (tmp_y_fore != 0).float()
                tmp_neg_mask = (tmp_y_fore == 0).float()
                sup_sem_feat_list = []
                b = tmp_mask.shape[0]  # batch size: fake num
                for i in range(b):
                    tmp_mask_bs = tmp_mask[i, :, :, :]
                    tmp_neg_mask_bs = tmp_neg_mask[i, :, :, :]
                    if torch.sum(tmp_mask_bs) != 0 and torch.sum(tmp_neg_mask_bs) != 0:
                        pos_proto = ((tmp_sem_sg_x[i, :, :, :] * tmp_mask_bs).sum(-1).sum(-1) /
                                     (tmp_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        neg_proto = ((tmp_sem_sg_x[i, :, :, :] * tmp_neg_mask_bs).sum(-1).sum(-1) /
                                     (tmp_neg_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        sup_sem_feat_list += [torch.cat([neg_proto, pos_proto], dim=0).unsqueeze(0)]
                ####################################################################################################

                ######################### generate centroid mask average pooling on bs channel #######################
                tmp_y_cen = F.interpolate(y_cen.float().unsqueeze(1), size=(h, w), mode='nearest')
                tmp_cen_mask = (tmp_y_cen != 0).float()
                tmp_neg_cen_mask = (tmp_y_cen == 0).float()
                sup_cen_feat_list = []
                b = tmp_cen_mask.shape[0]  # batch size: fake num
                for i in range(b):
                    tmp_cen_mask_bs = tmp_cen_mask[i, :, :, :]
                    tmp_neg_cen_mask_bs = tmp_neg_cen_mask[i, :, :, :]
                    if torch.sum(tmp_cen_mask_bs) != 0 and torch.sum(tmp_neg_cen_mask_bs) != 0:
                        pos_proto = ((tmp_cen_sg_x[i, :, :, :] * tmp_cen_mask_bs).sum(-1).sum(-1) /
                                     (tmp_cen_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        neg_proto = ((tmp_cen_sg_x[i, :, :, :] * tmp_neg_cen_mask_bs).sum(-1).sum(-1) /
                                     (tmp_neg_cen_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        sup_cen_feat_list += [torch.cat([neg_proto, pos_proto], dim=0).unsqueeze(0)]
                ######################################################################################################

                ######################### generate contour mask average pooling on bs channel #######################
                tmp_y_con = F.interpolate(y_con.float().unsqueeze(1), size=(h, w), mode='nearest')
                tmp_con_mask = (tmp_y_con != 0).float()
                tmp_neg_con_mask = (tmp_y_con == 0).float()
                sup_con_feat_list = []
                b = tmp_con_mask.shape[0]  # batch size: fake num
                for i in range(b):
                    tmp_con_mask_bs = tmp_con_mask[i, :, :, :]
                    tmp_neg_con_mask_bs = tmp_neg_con_mask[i, :, :, :]
                    if torch.sum(tmp_con_mask_bs) != 0 and torch.sum(tmp_neg_con_mask_bs) != 0:
                        pos_proto = ((tmp_con_sg_x[i, :, :, :] * tmp_con_mask_bs).sum(-1).sum(-1) /
                                     (tmp_con_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        neg_proto = ((tmp_con_sg_x[i, :, :, :] * tmp_neg_con_mask_bs).sum(-1).sum(-1) /
                                     (tmp_neg_con_mask_bs.sum(-1).sum(-1) + 1e-12)).unsqueeze(0)
                        sup_con_feat_list += [torch.cat([neg_proto, pos_proto], dim=0).unsqueeze(0)]
                ######################################################################################################


            return sup_sem_feat_list, \
                   sup_cen_feat_list, \
                   sup_con_feat_list,

        # get input size
        x_size = x.size()
        h = int(x_size[2])
        w = int(x_size[3])
        # input images to encoder
        d0, d1, d2, d3 = self.backbone(x, self.freeze)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]
        # define output dict
        out_dict = OrderedDict()

        if meta_train_model:
            # training by set fake support images`
            fake_num = x.size(0) // 2

            for branch_name, branch_desc in self.decoder.items():
                u3 = self.upsample2x(d[-1]) + d[-2]
                u3 = branch_desc[0](u3)
                u2 = self.upsample2x(u3) + d[-3]
                u2 = branch_desc[1](u2)
                u1 = self.upsample2x(u2) + d[-4]
                u1 = branch_desc[2](u1)

                # semantic seg branch: adding the semantic and centroid prototype guidance
                if branch_name == 'np':
                    sem_guid_u = self.semp_guided_conv(u1)
                    sup_batch_sem_feat_list = \
                        generate_sem_structure_proto(x=sem_guid_u[fake_num:],
                                                     y_fore=y_fore[fake_num:])

                    ################################## meta pred #######################################
                    sup_semantic_feat = torch.cat(sup_batch_sem_feat_list, dim=0).mean(0)
                    sem_meta_pred = self.get_pred(sem_guid_u, sup_semantic_feat)
                    ###########################################################################################


                # centroid seg branch: adding the centroid prototype guidance
                if branch_name == 'cenp':
                    cen_guid_u = self.cenp_guided_conv(u1)
                    sup_batch_cen_feat_list = \
                        generate_cen_structure_proto(x=cen_guid_u[fake_num:],
                                                    y_cen=y_cen[fake_num:])
                    ################################## meta pred #######################################
                    sup_cen_feat = torch.cat(sup_batch_cen_feat_list, dim=0).mean(0)
                    cen_meta_pred = self.get_pred(cen_guid_u, sup_cen_feat)
                    ###########################################################################################


                # contour seg branch: adding the contour prototype guidance
                if branch_name == 'conp':
                    con_guid_u = self.conp_guided_conv(u1)
                    sup_batch_con_feat_list = \
                        generate_con_structure_proto(x=con_guid_u[fake_num:],
                                                    y_con=y_con[fake_num:])
                    ################################## meta pred #######################################
                    sup_con_feat = torch.cat(sup_batch_con_feat_list, dim=0).mean(0)
                    con_meta_pred = self.get_pred(con_guid_u, sup_con_feat)
                    ###########################################################################################

                u0 = branch_desc[3](u1)
                out_dict[branch_name] = u0


            # meta pred
            con_meta_pred = F.interpolate(con_meta_pred, size=(h, w), mode='bilinear', align_corners=True)
            sem_meta_pred = F.interpolate(sem_meta_pred, size=(h, w), mode='bilinear', align_corners=True)
            cen_meta_pred = F.interpolate(cen_meta_pred, size=(h, w), mode='bilinear', align_corners=True)

            # momenton updata proto
            self._momentum_update_sg_memory_bank(sup_batch_sem_feat_list,
                                                 sup_batch_cen_feat_list,
                                                 sup_batch_con_feat_list)

            return out_dict, \
                   sem_meta_pred, \
                   cen_meta_pred, \
                   con_meta_pred, \

        else: # finetuning | evalution | evalution wo cls prototypes
            for branch_name, branch_desc in self.decoder.items():
                u3 = self.upsample2x(d[-1]) + d[-2]
                u3 = branch_desc[0](u3)
                u2 = self.upsample2x(u3) + d[-3]
                u2 = branch_desc[1](u2)
                u1 = self.upsample2x(u2) + d[-4]
                u1 = branch_desc[2](u1)

                # semantic seg branch: adding the semantic and centroid prototype guidance
                if branch_name == 'np':

                    if eval_with_sg:
                        sem_guid_u = self.semp_guided_conv(u1)

                        if eval_model and gened_sem_proto != None:
                            sem_cos_sim_map = self.get_pred(sem_guid_u, gened_sem_proto)
                        else:
                            sem_cos_sim_map = self.get_pred(sem_guid_u, self.semantic_proto_bank.detach())
                            fore_pred = sem_cos_sim_map[:, 1:, ...].mean(1).unsqueeze(1)
                            sem_cos_sim_map = torch.cat([sem_cos_sim_map[:, 0, ...].unsqueeze(1), fore_pred], dim=1)

                # centroid seg branch: adding the centroid prototype guidance
                if branch_name == 'cenp':

                    if eval_with_sg:
                        cen_guid_u = self.cenp_guided_conv(u1)
                        if eval_model and gened_cen_proto != None:
                            cen_cos_sim_map = self.get_pred(cen_guid_u, gened_cen_proto)
                        else:
                            cen_cos_sim_map = self.get_pred(cen_guid_u, self.centroid_proto_bank.detach())
                            fore_pred = cen_cos_sim_map[:, 1:, ...].mean(1).unsqueeze(1)
                            cen_cos_sim_map = torch.cat([cen_cos_sim_map[:, 0, ...].unsqueeze(1), fore_pred], dim=1)

                # contour seg branch: adding the contour prototype guidance
                if branch_name == 'conp':

                    if eval_with_sg:
                        cp_guid_u = self.conp_guided_conv(u1)
                        if eval_model and gened_con_proto != None:
                            con_cos_sim_map = self.get_pred(cp_guid_u, gened_con_proto)
                        else:
                            con_cos_sim_map = self.get_pred(cp_guid_u, self.contour_proto_bank.detach())
                            fore_pred = con_cos_sim_map[:, 1:, ...].mean(1).unsqueeze(1)
                            con_cos_sim_map = torch.cat([con_cos_sim_map[:, 0, ...].unsqueeze(1), fore_pred], dim=1)

                u0 = branch_desc[3](u1)
                out_dict[branch_name] = u0

            if eval_model:
                if eval_with_sg:
                    return out_dict, sem_cos_sim_map, cen_cos_sim_map, con_cos_sim_map
                else:
                    return out_dict



    def get_pred(self, x, proto):
        b, c, h, w = x.size()[:]
        if len(proto.shape[:]) == 3:
            # x: [b, c, h, w]
            # proto: [b, cls, c]
            cls_num = proto.size(1)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, -1, True)  # b, n, c
            x = x.contiguous().view(b, c, h * w)  # b, c, hw
            pred = proto @ x  # b, cls, hw
        elif len(proto.shape[:]) == 2:
            cls_num = proto.size(0)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, 1, True)
            x = x.contiguous().view(b, c, h * w)  # b, c, hw
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 10

    def get_weight_sg_proto(self, anchor_proto, proto_list, sim=True):
        """
        :param anchor_proto: bg/sem/con/cen prototype
        :param proto_list: prototype list for support
        :return: weighted proto
        """
        sup_feat = torch.cat(proto_list)
        sup_proto_norm = F.normalize(sup_feat, 2, -1)  # 1, c
        anchor_proto_norm = F.normalize(anchor_proto, 2, -1)  # bank_size, c

        # protos with higher/lower similarity with anchor proto has higher weight
        if sim:
            weight = (sup_proto_norm * anchor_proto_norm).sum(-1).unsqueeze(-1)
        else:
            weight = 1 - (sup_proto_norm * anchor_proto_norm).sum(-1).unsqueeze(-1)

        weight = weight * (weight > 0).float()
        weight = F.softmax(weight, dim=0)
        upgrade_sup_proto = (weight * sup_feat).sum(0).unsqueeze(0)
        return upgrade_sup_proto

    def combine_weight_sg_proto(self, sg_list, sg_bank):
        upgrade_feat_proto_list = []

        # bg proto
        sup_neg_feat_list = [feat[:, 0, ...] for feat in sg_list]
        upgrade_feat_proto_list += [self.get_weight_sg_proto(sg_bank.data[0], sup_neg_feat_list)]
        # fg proto
        sup_pos_feat_list = [feat[:, 1, ...] for feat in sg_list]
        for i in range(1, self.modal_num + 1):
            upgrade_feat_proto_list += [self.get_weight_sg_proto(sg_bank.data[i], sup_pos_feat_list)]

        assert len(upgrade_feat_proto_list) == self.modal_num + 1
        upgrade_proto = torch.cat(upgrade_feat_proto_list, dim=0)
        return upgrade_proto

    @torch.no_grad()
    def _momentum_update_sg_memory_bank(self, sup_sem_list, sup_cen_list, sup_con_list):
        """
        Momentum update of the structure prototype
        """
        keep_ratio = 0.95
        # sem bank update
        upgrade_semantic_proto = self.combine_weight_sg_proto(sup_sem_list, self.semantic_proto_bank)
        self.semantic_proto_bank.data.mul_(keep_ratio).add_(upgrade_semantic_proto.cuda(), alpha=1 - keep_ratio)
        # cen bank update
        upgrade_cen_proto = self.combine_weight_sg_proto(sup_cen_list, self.centroid_proto_bank)
        self.centroid_proto_bank.data.mul_(keep_ratio).add_(upgrade_cen_proto.cuda(), alpha=1 - keep_ratio)
        # con bank update
        upgrade_con_proto = self.combine_weight_sg_proto(sup_con_list, self.contour_proto_bank)
        self.contour_proto_bank.data.mul_(keep_ratio).add_(upgrade_con_proto.cuda(), alpha=1 - keep_ratio)
        return

    @torch.no_grad()
    def _sg_bank_registration(self, sup_sem_list, sup_cen_list, sup_con_list):
        """
        registration of structure guidance prototypes
        """
        refine_semantic_proto = torch.cat(sup_sem_list, dim=0).mean(0)
        refine_centroid_proto = torch.cat(sup_cen_list, dim=0).mean(0)
        refine_contour_proto = torch.cat(sup_con_list, dim=0).mean(0)
        return refine_semantic_proto, refine_centroid_proto, refine_contour_proto


if __name__ == '__main__':
    y = torch.randint(low=0, high=5, size=(4, 256, 256))
    x = F.one_hot(y, num_classes=6)
    z = 1
    # x = torch.randn(size=(1, 512))
    # y = torch.randn(size=(10, 512))
    # z = torch.concat([x.repeat(10, 1), y], dim=1)

    torch.cuda.set_device(1)
    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(0)
    model = SGFSL(modal_num=1).to(device)
    model.train()
    # model.eval()

    x = torch.randn(size=(4, 3, 256, 256))
    y = torch.randint(low=0, high=5, size=(4, 256, 256)).to(device)
    # y = torch.ones(size=(4, 256, 256))
    y_2 = torch.zeros(size=(4, 256, 256)).to(device)
    y_3 = torch.ones(size=(4, 256, 256)).to(device)
    y_4 = torch.ones(size=(4, 256, 256)).to(device)

    # x = torch.randn(size=(1, 1, 1, 3, 256, 256))
    # y = torch.ones(size=(1, 1, 1, 256, 256))

    y = model(x.to(device), y.to(device), y_2, y_3, y_4, meta_train_model=True)
