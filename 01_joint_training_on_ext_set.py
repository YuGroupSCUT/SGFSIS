import logging
import os
import random
import cv2
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
from skimage import morphology
from scipy.ndimage import binary_fill_holes
from scipy import ndimage
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from collections import OrderedDict
from models.post_proc import process, remove_small_objects
from models.SGFSL import SGFSL
from utils.loss import xentropy_loss, dice_loss, mse_loss
from dataset.nuclei_dataset import EXT_Metatrain_Dataset, QuadCompose, QuadRandomFlip, QuadRandomRotate
from utils import config
from utils.logger import get_logger
from utils.viz_utils import draw_overlay_scaling, draw_overlay_by_index
from utils.metrics import run_nuclei_type_stat
from utils.metrics import generate_cls_info
from utils.metrics import get_fast_aji, getmPQ
import sys

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

pannuke_col = np.array([[32, 32, 32],
                        [0, 0, 255],
                        [0, 255, 0],
                        [255, 0, 0],
                        [0, 255, 255],
                        [0, 165, 255]])

consep_col = np.array([[32, 32, 32],
                       [0, 255, 255],
                       [255, 0, 255],
                       [0, 0, 255],
                       [255, 0, 0]])

lizard_col = np.array([[32, 32, 32],
                       [0, 165, 255],
                       [0, 255, 0],
                       [0, 0, 255],
                       [255, 255, 0],
                       [255, 0, 0],
                       [0, 255, 255]])

monusac_col = np.array([[32, 32, 32],
                        [0, 0, 255],
                        [0, 255, 255],
                        [0, 255, 0],
                        [255, 0, 0]])

col_dict = {'pannuke': pannuke_col,
            'consep': consep_col,
            'monusac': monusac_col,
            'monusac_20x': monusac_col,
            'lizard': lizard_col}

loss_func_dict = {"bce": xentropy_loss,
                  "dice": dice_loss,
                  "con_bce": xentropy_loss,
                  "con_dice": dice_loss,
                  "cen_mse": mse_loss,
                  "type_bce": xentropy_loss,
                  'type_dice': dice_loss}

loss_opts = {"np": {"bce": 1, "dice": 1},
             "conp": {"con_bce": 1, "con_dice": 1},
             "cenp": {"cen_mse": 10},
             "tp": {"type_bce": 1, "type_dice": 1}}


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Nuclei Instance Segmentation')
    parser.add_argument('--config', type=str, default='./config/ext_joint_training/monusac_20x.yaml',
                        help='config file')
    parser.add_argument('opts', help='see .yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)
    np.random.seed(args.manual_seed + worker_id)


def main():
    args = get_parser()
    assert args.classes > 1
    torch.cuda.set_device(args.train_gpu[0])
    global device
    device = torch.device(f'cuda:{args.train_gpu[0]}' if torch.cuda.is_available() else 'cpu')

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
    main_worker(args)


def main_worker(argss):
    global args
    args = argss
    frozen_list = ['semantic_proto_bank', 'contour_proto_bank', 'centroid_proto_bank']

    model = SGFSL(num_types=args.classes, modal_num=args.modal_num)  # , criterion=criterion)
    for param in model.named_parameters():
        if param[0].split('.')[0] in frozen_list:
            print(param[0])
            param[1].requires_grad = False

    optimizer = torch.optim.Adam(
        [{'params': model.backbone.parameters()},
         {'params': model.conv_bot.parameters()},
         {'params': model.decoder.parameters()},
         {'params': model.main_proto},
         {'params': model.semp_guided_conv.parameters()},
         {'params': model.cenp_guided_conv.parameters()},
         {'params': model.conp_guided_conv.parameters()},
         ],
        lr=args.base_lr, weight_decay=args.weight_decay)

    save_path = os.path.join('./exp', 'ext_joint_training', f'{args.ext_source}',
                             f'{args.model}', 'model_weight')
    vis_save_dir = os.path.join('./exp', 'ext_joint_training', f'{args.ext_source}',
                                f'{args.model}', 'vis')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    global logger
    import datetime
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    logger = get_logger(os.path.join('./exp', 'ext_joint_training', f'{args.ext_source}',
                                     f'{args.model}',
                                     f'train_log_{cur_time}.txt'))

    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.train_gpu)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=args.milestones,
                                                     gamma=0.5)

    ############################# set dataloader ###################################
    ext_transform = QuadCompose([QuadRandomRotate(),
                                 QuadRandomFlip()])
    train_sampler = None
    train_data = EXT_Metatrain_Dataset(ext_root=args.ext_train_root,
                                       transform=ext_transform,
                                       mag=args.mag,
                                       dataset_name=args.ext_source)
    train_loader = torch.utils.data.DataLoader(train_data, worker_init_fn=worker_init_fn,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    test_data = EXT_Metatrain_Dataset(ext_root=args.ext_val_root,
                                      transform=None,
                                      mag=args.mag,
                                      dataset_name=args.ext_source)
    test_loader = torch.utils.data.DataLoader(test_data, worker_init_fn=worker_init_fn,
                                              batch_size=args.batch_size_val,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False)
    print(f'{len(train_data)} for train; {len(test_data)} for valid')

    # ############################## evaluate the performance ###################################
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(train_loader, model, optimizer, epoch, num_class=args.classes)
        scheduler.step()
        if epoch % 5 == 0:
            aji, F1_det, mpq, F_score, F1_avg = evaluate_one_epoch_patch_level(model,
                                                                               test_loader=test_loader,
                                                                               test_dataset=test_data,
                                                                               num_type=args.classes,
                                                                               dataset=args.ext_source,
                                                                               is_draw=True,
                                                                               save_dir=vis_save_dir)

            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}
            logger.info('Epoch: {}, aji: {}, mpq: {}, det f1 {}'.format(epoch, aji, mpq, F1_det))
            logger.info('f_score: {}'.format(F_score))
            torch.save(save_files,
                       "{}/model_{}_aji_{}_mpq_{}.pth".format(save_path, epoch, aji, mpq))

def train_one_epoch(train_loader, model, optimizer, epoch, num_class):
    model.train()
    accu_loss = {"sup": {"bce": 0,
                         "dice": 0,
                         "con_bce": 0,
                         "con_dice": 0,
                         "cen_mse": 0,
                         "type_bce": 0,
                         'type_dice': 0,
                         "sem_metabce": 0,
                         "con_metabce": 0,
                         "cen_metabce": 0,
                         "cls_metabce": 0,
                         },
                 "total": 0}

    # max_iter = args.epochs * len(train_loader)
    for i, (img_input, cls_target, fore_target, _,
            contour_target, center_target, center_gaussian_target) in enumerate(train_loader):

        img_input = img_input.cuda(non_blocking=True)
        cls_target = cls_target.cuda(non_blocking=True)
        fore_target = fore_target.cuda(non_blocking=True)
        contour_target = contour_target.cuda(non_blocking=True)
        center_target = center_target.cuda(non_blocking=True)
        center_gaussian_target = center_target.cuda(non_blocking=True)

        # modified the format of gt
        true_np = fore_target.type(torch.int64)
        true_np = F.one_hot(true_np, num_classes=2).type(torch.float32)

        true_cp = contour_target.type(torch.int64)
        true_cp = F.one_hot(true_cp, num_classes=2).type(torch.float32)

        true_tp = cls_target.type(torch.int64)
        true_tp = F.one_hot(true_tp, num_classes=num_class).type(torch.float32)

        # for meta-training
        true_center = center_target.type(torch.int64)
        true_center = F.one_hot(true_center, num_classes=2).type(torch.float32)

        true_center_gaussian = center_gaussian_target

        true_dict = {"np": true_np,
                     "conp": true_cp,
                     'tp': true_tp,
                     'cenp': true_center_gaussian
                     }

        pred_dict, \
        cls_meta_pred, \
        sem_meta_pred, \
        cen_meta_pred, \
        con_meta_pred = model(
            x=img_input,
            y_cls=cls_target,
            y_fore=fore_target,
            y_con=contour_target,
            y_cen=center_target,
            meta_train_model=True)

        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)
        pred_dict["conp"] = F.softmax(pred_dict["conp"], dim=-1)
        pred_dict["cenp"] = pred_dict["cenp"][:, :, :, 0]

        ########################## meta prediction ########################
        # meta pred
        cls_meta_pred = cls_meta_pred.permute(0, 2, 3, 1).contiguous()
        cls_meta_pred = F.softmax(cls_meta_pred, dim=-1)

        con_meta_pred = con_meta_pred.permute(0, 2, 3, 1).contiguous()
        con_meta_pred = F.softmax(con_meta_pred, dim=-1)

        sem_meta_pred = sem_meta_pred.permute(0, 2, 3, 1).contiguous()
        sem_meta_pred = F.softmax(sem_meta_pred, dim=-1)

        cen_meta_pred = cen_meta_pred.permute(0, 2, 3, 1).contiguous()
        cen_meta_pred = F.softmax(cen_meta_pred, dim=-1) # [..., 1]
        ###################################################################

        loss = 0
        # 计算有监督部分的损失函数
        for branch_name in pred_dict.keys():
            for loss_name, loss_weight in loss_opts[branch_name].items():
                loss_func = loss_func_dict[loss_name]
                loss_args = [true_dict[branch_name], pred_dict[branch_name]]
                term_loss = loss_func(*loss_args) * loss_weight
                accu_loss["sup"][loss_name] += term_loss.cpu().item()
                if not torch.isfinite(term_loss.cpu()):  # 当计算的损失为无穷大时
                    logging.info("{} {} Loss is {}".format(branch_name, loss_name, term_loss))
                loss += term_loss

        cls_meta_loss = loss_func_dict['bce'](true_dict['tp'], cls_meta_pred) + \
                        loss_func_dict['dice'](true_dict['tp'], cls_meta_pred)
        sem_meta_loss = loss_func_dict['bce'](true_dict['np'], sem_meta_pred, true_dict['np'][..., 1]) + \
                        loss_func_dict['dice'](true_dict['np'], sem_meta_pred)
        con_meta_loss = loss_func_dict['bce'](true_dict['conp'], con_meta_pred, true_dict['conp'][..., 1]) + \
                        loss_func_dict['dice'](true_dict['conp'], con_meta_pred)
        cen_meta_loss = loss_func_dict['bce'](true_center, cen_meta_pred, true_dict['np'][..., 1]) + \
                        loss_func_dict['dice'](true_center, cen_meta_pred)

        loss = loss + cls_meta_loss + sem_meta_loss + cen_meta_loss + con_meta_loss

        # meta loss
        accu_loss["sup"]['cls_metabce'] += cls_meta_loss.cpu().item()
        accu_loss["sup"]['sem_metabce'] += sem_meta_loss.cpu().item()
        accu_loss["sup"]['cen_metabce'] += cen_meta_loss.cpu().item()
        accu_loss["sup"]['con_metabce'] += con_meta_loss.cpu().item()

        accu_loss["total"] += loss.cpu().item()

        if not torch.isfinite(loss.cpu()):  # 当计算的损失为无穷大时停止训练
            logging.info("Loss is {}, stopping training".format(loss))
            print(cls_meta_loss.cpu().item(), sem_meta_loss.cpu().item(),
                  cen_meta_loss.cpu().item())
            # return
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info(
        "Epoch: [{}/{}][{}/{}]  loss:{:.3f}  "
        "bce:{:.3f}  dice:{:.3f}  "
        "con_bce:{:.3f}  con_dice:{:.3f}  "
        "cen_mse: {:.3f}  "
        "type_bce:{:.3f} type_dice:{:.3f} "
        "sem_metabce:{:.3f}  cen_metabce:{:.3f} con_metabce:{:.3f} cls_metabce{:.3f}  "
        "lr:{:.6f}".format(
            epoch, args.epochs, 1, 1,
            accu_loss["total"] / len(train_loader),
            accu_loss["sup"]["bce"] / len(train_loader),
            accu_loss["sup"]["dice"] / len(train_loader),
            accu_loss["sup"]["con_bce"] / len(train_loader),
            accu_loss["sup"]["con_dice"] / len(train_loader),
            accu_loss["sup"]["cen_mse"] / len(train_loader),
            accu_loss["sup"]["type_bce"] / len(train_loader),
            accu_loss["sup"]["type_dice"] / len(train_loader),
            accu_loss["sup"]["sem_metabce"] / len(train_loader),
            accu_loss["sup"]["cen_metabce"] / len(train_loader),
            accu_loss["sup"]["con_metabce"] / len(train_loader),
            accu_loss["sup"]["cls_metabce"] / len(train_loader),
            optimizer.state_dict()['param_groups'][0]['lr']
        )
    )

    return


def evaluate_one_epoch_patch_level(model,
                                   test_loader,
                                   test_dataset,
                                   num_type=5,
                                   device='cuda',
                                   dataset='',
                                   is_draw=True,
                                   num_run=1,
                                   save_dir=''):
    vis_save_dir = save_dir
    os.makedirs(vis_save_dir, exist_ok=True)
    model.eval()

    results = []
    for i, (img_input, cls_target, fore_target, inst_target, _, _, _) in enumerate(test_loader):

        true_tp = cls_target.numpy()
        true_inst = inst_target.numpy()

        with torch.no_grad():
            pred_dict_, sem_sim_map, cen_sim_map, con_sim_map = \
                model(x=img_input.to(device),
                      gen_proto=False,
                      eval_model=False,
                      eval_model_wo_proto=True,
                      eval_with_sg=True,
                      gened_proto=False)

        sem_sim_map = sem_sim_map.permute(0, 2, 3, 1).contiguous()
        sem_sim_map = F.softmax(sem_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]
        cen_sim_map = cen_sim_map.permute(0, 2, 3, 1).contiguous()
        cen_sim_map = F.softmax(cen_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]
        con_sim_map = con_sim_map.permute(0, 2, 3, 1).contiguous()
        con_sim_map = F.softmax(con_sim_map, dim=-1)[..., 1].detach().cpu().numpy()[0]

        # confirm the branch prediction output order np-cp-tp
        pred_dict = OrderedDict()
        pred_dict['np'] = pred_dict_['np']
        pred_dict['conp'] = pred_dict_['conp']
        pred_dict['cenp'] = pred_dict_['cenp']
        pred_dict['tp'] = pred_dict_['tp']

        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        pred_dict["conp"] = F.softmax(pred_dict["conp"], dim=-1)[..., 1:]

        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

        # x = pred_output.detach().cpu().numpy()
        pred_inst, inst_info_dict = process(pred_output.detach().cpu().numpy()[0],
                                            nr_types=num_type,
                                            sem_guid_map=sem_sim_map,
                                            con_guid_map=con_sim_map,
                                            cen_guid_map=cen_sim_map,
                                            )

        results += [(true_inst[0], true_tp[0],
                     pred_output.detach().cpu().numpy()[0],
                     pred_inst, inst_info_dict,
                     sem_sim_map, cen_sim_map, con_sim_map)]

    ############################ evalute the performence #################################
    total_aji = 0
    list_length = len(results)
    res = []
    true_res = []
    pred_dict = {}
    true_dict = {}
    for idx, result in enumerate(results):
        ori_img = cv2.imread(test_dataset.img_list[idx])
        img_name = os.path.basename(test_dataset.img_list[idx]).split('.')[0]
        inst_label, cls_label, pred_output, pred_inst, inst_info_dict, \
        sem_sim_map, cen_sim_map, con_sim_map = result
        # cls to be ingnored in gt labels (0 means ignore)
        ambiguous_mask = np.ones_like(inst_label).astype(np.bool)

        if dataset == 'monusac' or dataset == 'monusac_20x':
            # order can not be changed
            ambiguous_mask = (cls_label != 5)
            inst_label *= ambiguous_mask
            cls_label *= ambiguous_mask

        pred_cls = np.zeros_like(pred_inst)
        if inst_info_dict is not None:
            for inst in range(1, int(np.max(pred_inst)) + 1):
                try:
                    pred_cls[pred_inst == inst] = inst_info_dict[inst]['type']
                except:
                    pass

        # set the cls as 0
        pred_cls *= ambiguous_mask
        pred_inst *= ambiguous_mask

        pred_dict[img_name] = generate_cls_info(pred_inst, pred_cls)
        true_dict[img_name] = generate_cls_info(inst_label, cls_label)

        res += [np.concatenate([pred_inst[..., None], pred_cls[..., None]], axis=-1)[None, ...]]
        true_res += [np.concatenate([inst_label[..., None], cls_label[..., None]], axis=-1)[None, ...]]

        # draw_overlay
        if is_draw and num_run == 1:

            gt_vis = draw_overlay_scaling(ori_img, inst_label, cls_label, col_dict[dataset])
            pred_vis = draw_overlay_scaling(ori_img, pred_inst, pred_cls, col_dict[dataset])

            pred_np = pred_output[:, :, 0]
            pred_np = 255 * np.array(pred_np >= 0.5, dtype=np.int32)
            pred_np = pred_np.astype(np.uint8)
            pred_np = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)

            pred_conp = pred_output[:, :, 1]
            pred_conp = 255 * np.array(pred_conp >= 0.5, dtype=np.int32)
            pred_conp = pred_conp.astype(np.uint8)
            pred_conp = cv2.cvtColor(pred_conp, cv2.COLOR_GRAY2BGR)

            pred_cenp = pred_output[:, :, 2]
            pred_cenp = 255 * np.array(pred_cenp >= 0.5, dtype=np.int32)
            pred_cenp = pred_cenp.astype(np.uint8)
            pred_cenp = cv2.cvtColor(pred_cenp, cv2.COLOR_GRAY2BGR)

            pred_type = pred_output[..., -1]
            pred_type = draw_overlay_by_index(pred_type, col_dict[dataset])

            # vis_guidance
            # sem_guid_map = np.array(sem_sim_map >= 0.5, dtype=np.int32)
            # con_guid_map = np.array(con_sim_map >= 0.5, dtype=np.int32)
            # con_guid_map = np.array(cen_sim_map >= 0.5, dtype=np.int32)

            norm_img = cv2.normalize(sem_sim_map, None, 32, 224, cv2.NORM_MINMAX)
            norm_img = np.asarray(norm_img, dtype=np.uint8)
            sem_sim_heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

            norm_img = cv2.normalize(cen_sim_map, None, 32, 224, cv2.NORM_MINMAX)
            norm_img = np.asarray(norm_img, dtype=np.uint8)
            cen_sim_heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

            edge1 = np.ones(shape=(pred_vis.shape[0], 5, 3)) * 255
            img_2_save = np.concatenate([
                gt_vis, edge1,
                pred_vis, edge1,
                pred_type, edge1,
                pred_np, edge1,
                pred_conp, edge1,
                pred_cenp, edge1,
                sem_sim_heat_img, edge1,
                cen_sim_heat_img
            ],
                axis=1)
            cv2.imwrite(f'{vis_save_dir}/{img_name}.png', img_2_save)

        # if empty test patch, ignore the metric
        if np.max(inst_label) == 0 and np.max(pred_inst) == 0:
            aji = 0
            list_length -= 1
        elif np.max(inst_label) != 0 and np.max(pred_inst) == 0:
            aji = 0
        else:
            aji = get_fast_aji(inst_label, pred_inst)
        total_aji += aji

    pq_metrics = getmPQ(res, true_res, nr_classes=num_type - 1)
    mpq = pq_metrics['multi_pq+'][0]
    print("mpq: ", mpq)

    # classification metrics
    if dataset == 'consep_20x' or dataset == 'monusac_20x' or dataset == 'lizard':
        _20x = True
    else:
        _20x = False
    results_list = run_nuclei_type_stat(pred_dict, true_dict, _20x=_20x)
    F1_det = results_list[0]
    F1_list = results_list[-int(num_type - 1):]
    F1_avg = np.mean(F1_list)

    return total_aji / list_length, F1_det, mpq, F1_list, F1_avg


if __name__ == '__main__':
    main()