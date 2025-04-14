import logging
import os
import random
import cv2
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from scipy import ndimage
import torch.nn.functional as F
from collections import OrderedDict
from models.post_proc import process, remove_small_objects
from models.SGFSL import SGFSL
from utils.loss import xentropy_loss, dice_loss, mse_loss
from dataset.nuclei_dataset import Tsk_Finetune_Dataset, QuadCompose, QuadRandomFlip, QuadRandomRotate
from utils import config
from utils.logger import get_logger
from utils.viz_utils import draw_overlay_scaling, draw_overlay_by_index
from utils.metrics import run_nuclei_type_stat
from utils.metrics import generate_cls_info
from utils.metrics import get_fast_aji, getmPQ, get_dice

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
    parser.add_argument('--config', type=str,
                        default='./config/tsk_finetuning/ext_monusac_tsk_consep.yaml',
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

def model_init():
    frozen_list = ['semantic_proto_bank', 'contour_proto_bank', 'centroid_proto_bank']
    model = SGFSL(num_types=args.classes, modal_num=args.modal_num)
    for param in model.named_parameters():
        if param[0].split('.')[0] in frozen_list:
            print('frozen layers:', param[0])
            param[1].requires_grad = False

    optimizer = torch.optim.Adam(
        [{'params': model.backbone.parameters()},
         {'params': model.decoder['tp'].parameters(), 'lr': args.base_lr * 10},
         {'params': model.decoder['np'].parameters()},
         {'params': model.decoder['conp'].parameters()},
         {'params': model.decoder['cenp'].parameters()},
         {'params': model.conv_bot.parameters()},
         {'params': model.main_proto, 'lr': args.base_lr * 10},
         {'params': model.semp_guided_conv.parameters()},
         {'params': model.cenp_guided_conv.parameters()},
         {'params': model.conp_guided_conv.parameters()}
         ],
        lr=args.base_lr, weight_decay=args.weight_decay)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.train_gpu)

    ####################################### TODO: load pretrained model #################################
    model_weight = torch.load(args.pretrained_weight, map_location='cpu')['model']

    load_weight_dict = {k: v for k, v in model_weight.items()
                        if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weight_dict, strict=False)

    # TODO: replace the classification prototypes
    with torch.no_grad():
        model.module.main_proto = nn.Parameter(torch.randn(args.classes, 512).cuda())

    # # for monusac -> pannuke
    # with torch.no_grad():
    #     model.module.main_proto[:3] = nn.Parameter(model_weight['module.main_proto'][:3].cuda())

    # # for consep -> monusac
    # with torch.no_grad():
    #     model.module.main_proto[0] = nn.Parameter(model_weight['module.main_proto'][0].cuda())  # back --> back
    #     model.module.main_proto[2] = nn.Parameter(model_weight['module.main_proto'][2].cuda())  # # inf
    #     model.module.main_proto[1] = nn.Parameter(model_weight['module.main_proto'][3].cuda())  # # epi

    # for monusac -> consep
    with torch.no_grad():
        model.module.main_proto[0] = nn.Parameter(model_weight['module.main_proto'][0].cuda())  # back --> back
        model.module.main_proto[2] = nn.Parameter(model_weight['module.main_proto'][2].cuda())  # inf
        model.module.main_proto[3] = nn.Parameter(model_weight['module.main_proto'][1].cuda())  # epi

    # # monusac -> lizard
    # with torch.no_grad():
    #     model.module.main_proto[0] = nn.Parameter(model_weight['module.main_proto'][0].cuda())  # back --> back
    #     model.module.main_proto[1] = nn.Parameter(model_weight['module.main_proto'][4].cuda())  # neut
    #     model.module.main_proto[2] = nn.Parameter(model_weight['module.main_proto'][1].cuda())  # epi
    #     model.module.main_proto[3] = nn.Parameter(model_weight['module.main_proto'][2].cuda())  # lym

    return model, optimizer

def main():
    args = get_parser()
    assert args.classes > 1
    torch.cuda.set_device(args.train_gpu[0])
    global device

    device = torch.device(f'cuda:{args.train_gpu[0]}' if torch.cuda.is_available() else 'cpu')
    print(device)

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

    save_path = os.path.join('./exp', 'tsk_finetuning', f'exttrain_{args.ext_source}_tskfinetuning_{args.tsk_source}',
                             f'{args.model}', f'{args.label_mode}_{args.shot}', 'model_weight')
    regis_vis_save_dir = os.path.join('./exp', 'tsk_finetuning', f'exttrain_{args.ext_source}_tskfinetuning_{args.tsk_source}',
                                      f'{args.model}', f'{args.label_mode}_{args.shot}', 'regis_vis_kr0.0')
    ft_vis_save_dir = os.path.join('./exp', 'tsk_finetuning', f'exttrain_{args.ext_source}_tskfinetuning_{args.tsk_source}',
                                   f'{args.model}', f'{args.label_mode}_{args.shot}', 'finetuning_vis')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(regis_vis_save_dir, exist_ok=True)
    os.makedirs(ft_vis_save_dir, exist_ok=True)

    global logger
    import datetime
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    logger = get_logger(os.path.join('./exp', 'tsk_finetuning', f'exttrain_{args.ext_source}_tskfinetuning_{args.tsk_source}',
                                     f'{args.model}',
                                     f'{args.label_mode}_{args.shot}',
                                     f'train_log_{cur_time}.txt'))

    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    ############################# define test set dataloader ##############################

    test_data = Tsk_Finetune_Dataset(tsk_root=args.tsk_test_root,
                                     tsk_source=args.tsk_source,
                                     mode='val',
                                     transform=None,
                                     mag=args.mag,
                                     )

    test_loader = torch.utils.data.DataLoader(test_data, worker_init_fn=worker_init_fn,
                                              batch_size=args.batch_size_val,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False)

    print(f'{len(test_data)} samples for testing')

    tsk_loader_list = []
    tsk_loader_regis_list = []
    for tsk_supp_run in args.task_run_list:
    # for tsk_supp_run in [2, 3, 4, 5]:
        os.makedirs(os.path.join(save_path, f'run_{tsk_supp_run}'), exist_ok=True)
        print('processing split run: ', tsk_supp_run)
        tsk_transform = QuadCompose([QuadRandomRotate(),
                                     QuadRandomFlip()])


        tsk_supp_data = Tsk_Finetune_Dataset(mode='train',
                                             tsk_root=args.tsk_train_root,
                                             tsk_source=args.tsk_source,
                                             shot=args.shot,
                                             run=tsk_supp_run,
                                             transform=tsk_transform,
                                             mag=args.mag
                                             )

        tsk_supp_data_regis = Tsk_Finetune_Dataset(
            mode='train',
            tsk_root=args.tsk_train_root,
            tsk_source=args.tsk_source,
            shot=args.shot,
            run=tsk_supp_run,
            transform=None,
            mag=args.mag
        )


        tsk_supp_loader = torch.utils.data.DataLoader(tsk_supp_data,
                                                      worker_init_fn=worker_init_fn,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.workers,
                                                      pin_memory=True,
                                                      sampler=None)

        # if shot > 10, support sample can not be sent to network restricted to gpu memory
        if args.shot < 10:
            regis_batch_size = args.tsk_num * args.shot
        else:
            regis_batch_size = 20

        tsk_supp_loader_regis = torch.utils.data.DataLoader(
            tsk_supp_data_regis,
            worker_init_fn=worker_init_fn,
            batch_size=regis_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None)

        tsk_loader_list.append(tsk_supp_loader)
        tsk_loader_regis_list.append(tsk_supp_loader_regis)

        print(f'num of {tsk_supp_run}: {len(tsk_supp_data)}')

    aji_list = []
    mpq_list = []
    pq_list = []
    dice_list = []
    f1_det_list = []
    f1_cls_list = []
    f1_base_list = []
    f1_novel_list = []
    multi_run_mean_f_score = np.zeros(shape=(args.classes - 1))
    for run, (tsk_loader, tsk_loader_regis) in enumerate(zip(tsk_loader_list, tsk_loader_regis_list), start=1):

        logger.info(f'################################## start run {run} #####################################')

        model, optimizer = model_init()
        gened_proto, _, _, _ = get_new_proto(tsk_loader_regis,
                                             model,
                                             novel_list=args.novel_list,
                                             base_list=args.base_list)
        model.module._cls_bank_registration(gened_proto)

        logger.info('####################### eval w finetuning run {} #####################'.format(run))

        #########################init with new prototypes for finetuning #####################################
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
        #                                                  milestones=args.milestones,
        #                                                  gamma=0.5)

        for epoch in range(args.start_epoch, args.epochs[run - 1] + 1):
            train_one_epoch(tsk_loader, model, optimizer, epoch, num_class=args.classes, run=run)
            # scheduler.step()

            if epoch % 50 == 0 or epoch % (args.epochs[run - 1]) == 0:
            # if epoch % (args.epochs[run - 1]) == 0:
                logger.info('####################### eval w regis run {} #####################'.format(run))
                gened_proto, \
                sup_sem_feat_list, \
                sup_cen_feat_list, \
                sup_con_feat_list = get_new_proto(tsk_loader_regis,
                                                  model,
                                                  novel_list=args.novel_list,
                                                  base_list=args.base_list)
                gened_sem_proto, \
                gened_cen_proto, \
                gened_con_proto = model.module._sg_bank_registration(sup_sem_feat_list,
                                                                     sup_cen_feat_list,
                                                                     sup_con_feat_list)

                aji, F1_det, mpq, F_score, F1_avg, pq, dice = evaluate_one_epoch_patch_level(model,
                                                                                   gened_proto=None,
                                                                                   gened_sem_proto=gened_sem_proto,
                                                                                   gened_cen_proto=gened_cen_proto,
                                                                                   gened_con_proto=gened_con_proto,
                                                                                   novel_list=args.novel_list,
                                                                                   base_list=args.base_list,
                                                                                   test_loader=test_loader,
                                                                                   test_dataset=test_data,
                                                                                   num_type=args.tsk_num + 1,
                                                                                   dataset=args.tsk_source,
                                                                                   is_draw=True,
                                                                                   save_dir=regis_vis_save_dir,
                                                                                   num_run=run)
                # aji, F1_det, mpq, F_score, F1_avg, pq, dice = evaluate_one_epoch_patch_level(model,
                #                                                                              gened_proto=None,
                #                                                                              gened_sem_proto=None,
                #                                                                              gened_cen_proto=None,
                #                                                                              gened_con_proto=None,
                #                                                                              novel_list=args.novel_list,
                #                                                                              base_list=args.base_list,
                #                                                                              test_loader=test_loader,
                #                                                                              test_dataset=test_data,
                #                                                                              num_type=args.tsk_num + 1,
                #                                                                              dataset=args.tsk_source,
                #                                                                              is_draw=True,
                #                                                                              save_dir=regis_vis_save_dir,
                #                                                                              num_run=run)
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}
                logger.info('Epoch: {}, aji: {}, mpq: {}, det f1 {}'.format(epoch, aji, mpq, F1_det))
                logger.info('f_score: {}'.format(F_score))
                torch.save(save_files,
                           "{}/run_{}/model_{}_aji_{}_mpq_{}.pth".format(save_path, run, epoch, aji, mpq))

        aji_list += [aji]
        mpq_list += [mpq]
        f1_det_list += [F1_det]
        f1_cls_list += [F1_avg]
        pq_list += [pq]
        dice_list += [dice]
        F_score = np.array(F_score).astype(np.float64)
        base_list = [i - 1 for i in args.base_list]
        f1_base_list += [np.mean(F_score[base_list])]
        novel_list = [i - 1 for i in args.novel_list]
        f1_novel_list += [np.mean(F_score[novel_list])]
        multi_run_mean_f_score += F_score

    logger.info('aji_mean_std: {}_{}'.format(np.mean(aji_list), np.std(aji_list)))
    logger.info('pq_mean_std: {}_{}'.format(np.mean(pq_list), np.std(pq_list)))
    logger.info('dice_mean_std: {}_{}'.format(np.mean(dice_list), np.std(dice_list)))
    logger.info('detection_f_score_mean_std: {}_{}'.format(np.mean(f1_det_list), np.std(f1_det_list)))
    logger.info('mpq_mean_std: {}_{}'.format(np.mean(mpq_list), np.std(mpq_list)))
    logger.info('classification_f_score_mean_std: {}_{}'.format(np.mean(f1_cls_list), np.std(f1_cls_list)))
    logger.info('classification_f_score_base_mean_std: {}_{}'.format(np.mean(f1_base_list), np.std(f1_base_list)))
    logger.info('classification_f_score_novel_mean_std: {}_{}'.format(np.mean(f1_novel_list), np.std(f1_novel_list)))

    multi_run_mean_f_score /= len(args.task_run_list)
    multi_run_mean_f_score = "F1_socre:" + "/".join("{}".format(f) for f in multi_run_mean_f_score)
    logger.info('mpq_mean_f1: {}'.format(multi_run_mean_f_score))


def get_new_proto(val_supp_loader, model, novel_list, base_list):
    logger.info('>>>>>>>>>>>>>>>> Start New Proto Generation >>>>>>>>>>>>>>>>')
    model.eval()

    # if shot > 10, support sample can not be sent to network restricted to gpu memory
    if args.shot < 10:
        n_size = args.tsk_num * args.shot
    else:
        n_size = 20

    # new_proto_num_epoch = 1  # 1
    with torch.no_grad():
        main_size = 512

        sup_sem_feat_list_bed = []
        sup_con_feat_list_bed = []
        sup_cen_feat_list_bed = []

        gened_proto_bed = torch.zeros(args.classes, main_size).cuda()

        # for epoch in range(new_proto_num_epoch):
        for i, (input, target, _, _, contour_target, centroid_target, _) in enumerate(val_supp_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            contour_target = contour_target.cuda(non_blocking=True)
            # logger.info('Generating new prototypes {}/{}...'.format(epoch, new_proto_num_epoch))
            logger.info('base_num: {}, novel_num: {}'.format(len(base_list), len(novel_list)))
            logger.info('Input: {}, Target: {}.'.format(input.shape, target.shape))

            input = input.contiguous().view(1, n_size, input.size(1), input.size(2), input.size(3))
            target = target.contiguous().view(1, n_size, target.size(1), target.size(2))
            contour_target = contour_target.contiguous().view(1, n_size, contour_target.size(1),
                                                              contour_target.size(2))
            centroid_target = centroid_target.contiguous().view(1, n_size, centroid_target.size(1),
                                                            centroid_target.size(2))
            input = input.repeat(8, 1, 1, 1, 1)
            target = target.repeat(8, 1, 1, 1)
            contour_target = contour_target.repeat(8, 1, 1, 1)
            centroid_target = centroid_target.repeat(8, 1, 1, 1)

            gened_proto, \
            sup_sem_feat_list, \
            sup_cen_feat_list, \
            sup_con_feat_list = model(x=input,
                                      y_cls=target,
                                      y_con=contour_target,
                                      y_cen=centroid_target,
                                      gen_proto=True,
                                      novel_list=novel_list,
                                      base_list=base_list)
            gened_proto = gened_proto.mean(0)
            gened_proto_bed = gened_proto_bed + gened_proto
            # np
            sup_sem_feat_list_bed += sup_sem_feat_list
            # cenp
            sup_cen_feat_list_bed += sup_cen_feat_list
            # cp
            sup_con_feat_list_bed += sup_con_feat_list

        gened_proto = gened_proto_bed / len(val_supp_loader)
        return gened_proto, \
               sup_sem_feat_list_bed, \
               sup_cen_feat_list_bed,\
               sup_con_feat_list_bed


def train_one_epoch(train_loader, model, optimizer, epoch, num_class, run):
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
        # cls_meta_pred = cls_meta_pred.permute(0, 2, 3, 1).contiguous()
        # cls_meta_pred = F.softmax(cls_meta_pred, dim=-1)

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

        # cls_meta_loss = loss_func_dict['bce'](true_dict['tp'], cls_meta_pred) + \
        #                 loss_func_dict['dice'](true_dict['tp'], cls_meta_pred)
        sem_meta_loss = loss_func_dict['bce'](true_dict['np'], sem_meta_pred, true_dict['np'][..., 1]) + \
                        loss_func_dict['dice'](true_dict['np'], sem_meta_pred)
        con_meta_loss = loss_func_dict['bce'](true_dict['conp'], con_meta_pred, true_dict['conp'][..., 1]) + \
                        loss_func_dict['dice'](true_dict['conp'], con_meta_pred)
        cen_meta_loss = loss_func_dict['bce'](true_center, cen_meta_pred, true_dict['np'][..., 1]) + \
                        loss_func_dict['dice'](true_center, cen_meta_pred)

        loss = loss + sem_meta_loss + cen_meta_loss + con_meta_loss # + cls_meta_loss

        # meta loss
        accu_loss["sup"]['cls_metabce'] += 0 # cls_meta_loss.cpu().item()
        accu_loss["sup"]['sem_metabce'] += sem_meta_loss.cpu().item()
        accu_loss["sup"]['cen_metabce'] += cen_meta_loss.cpu().item()
        accu_loss["sup"]['con_metabce'] += con_meta_loss.cpu().item()

        accu_loss["total"] += loss.cpu().item()

        # if not torch.isfinite(loss.cpu()):  # 当计算的损失为无穷大时停止训练
        #     logging.info("Loss is {}, stopping training".format(loss))
        #     print(cls_meta_loss.cpu().item(), sem_meta_loss.cpu().item(),
        #           cen_meta_loss.cpu().item())
        #     # return
        #     sys.exit(1)

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
            epoch, args.epochs[run - 1], 1, 1,
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

def rand_color_visualization(inst_mask):
    mask = np.zeros(shape=(256, 256, 3))

    ind_list = np.unique(inst_mask).tolist()
    ind_list.remove(0)
    for i in ind_list:
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        mask[inst_mask == i] = np.array([B, G, R])
    return mask

def evaluate_one_epoch_patch_level(model,
                                   gened_proto,
                                   novel_list,
                                   base_list,
                                   test_loader,
                                   test_dataset,
                                   gened_sem_proto=None,
                                   gened_cen_proto=None,
                                   gened_con_proto=None,
                                   num_type=5,
                                   device='cuda',
                                   dataset='',
                                   is_draw=True,
                                   save_dir='',
                                   num_run=1):

    eval_with_pt = True if gened_proto != None else False
    eval_without_pt = not eval_with_pt

    vis_save_dir = save_dir # os.path.join(save_dir, 'vis')
    os.makedirs(vis_save_dir, exist_ok=True)
    model.eval()
    results = []
    for i, (img_input, cls_target, fore_target, inst_target,
            contour_target, centroid_target, _) in enumerate(test_loader):

        true_tp = cls_target.numpy()
        true_inst = inst_target.numpy()

        with torch.no_grad():
            pred_dict_, sem_sim_map, cen_sim_map, con_sim_map = \
                model(x=img_input.to(device),
                       eval_model=eval_with_pt,
                       eval_model_wo_proto=eval_without_pt,
                       eval_with_sg=True,
                       gened_proto=gened_proto,
                       gened_sem_proto=gened_sem_proto,
                       gened_cen_proto=gened_cen_proto,
                       gened_con_proto=gened_con_proto,
                       novel_list=novel_list,
                       base_list=base_list,
                       )

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
    total_dice = 0
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

        if dataset == 'monusac':
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
            dice = 0
            list_length -= 1
        elif np.max(inst_label) != 0 and np.max(pred_inst) == 0:
            aji = 0
            dice = 0
        else:
            aji = get_fast_aji(inst_label, pred_inst)
            dice = get_dice(inst_label, pred_inst)

        total_aji += aji
        total_dice += dice

    pq_metrics = getmPQ(res, true_res, nr_classes=num_type - 1)
    pq = pq_metrics['pq']
    mpq = pq_metrics['multi_pq+'][0]
    # print("mpq: ", mpq)

    # classification metrics
    if dataset == 'consep_20x' or dataset == 'monusac_20x' or dataset == 'lizard':
        _20x = True
    else:
        _20x = False
    results_list = run_nuclei_type_stat(pred_dict, true_dict, _20x=_20x)
    F1_det = results_list[0]
    F1_list = results_list[-int(num_type - 1):]
    F1_avg = np.mean(F1_list)

    return total_aji / list_length, F1_det, mpq, F1_list, F1_avg, pq, total_dice / list_length


if __name__ == '__main__':
    main()
