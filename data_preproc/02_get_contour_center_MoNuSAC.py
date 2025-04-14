"""
split the patch 2 diff classes
"""
import copy
import glob
import os
import numpy as np
import cv2
import tqdm
import scipy.io as scio
from scipy import stats

# consep
# colorDict = {1: [0, 255, 255],
#              2: [255, 0, 255],
#              3: [0, 255, 0],
#              4: [0, 0, 255],
#              5: [255, 0, 0]}
#
# typesDict = {1: 'miscellaneous',
#              2: 'inflammatory',
#              3: 'healthy_epithelial',
#              4: 'malignant_epithelial',
#              5: 'spindle_shaped',
#              6: 'empty'}


if __name__ == '__main__':
    fold = 'test'
    patch_paths = os.listdir(f'/home/data1/my/dataset/monusac/extracted_mirror_20x/{fold}/256x256_256x256/np_files/')
    # monusac
    save_dir = f'/home/data1/my/dataset/monusac/extracted_mirror_20x/{fold}/256x256_256x256/'

    center_rad = 1
    contour_th = 1
    os.makedirs(os.path.join(save_dir, f'Centers_{center_rad}_{contour_th}'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, f'Contours_{center_rad}_{contour_th}'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, f'Label_cr_{center_rad}_ct_{contour_th}'), exist_ok=True)

    for patch_path in tqdm.tqdm(patch_paths):
        # monusac
        save_name = patch_path.split('.')[0]
        basePath = f'/home/data1/my/dataset/monusac/extracted_mirror_20x/{fold}/256x256_256x256/'
        img = cv2.imread(os.path.join(basePath, 'Images', save_name + '.png'))

        inst_map = scio.loadmat(os.path.join(basePath, 'Labels',save_name + '.mat'))["inst_map"]
        cls_map = scio.loadmat(os.path.join(basePath, 'Labels', save_name + '.mat'))["cls_map"]

        # get the nuclei instance
        insts = np.unique(inst_map).tolist()
        insts.remove(0)

        # draw contour
        img_center = np.zeros(shape=(256, 256)).astype(np.uint8)
        img_contour = np.zeros(shape=(256, 256)).astype(np.uint8)
        for ins in insts:
            inst_mask = copy.deepcopy(inst_map)
            # get the inst class
            cls_num = int(np.unique(cls_map[inst_mask == ins]))

            inst_mask[inst_mask != ins] = 0
            inst_mask[inst_mask > 0] = 255

            inst_mask = inst_mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in range(len(contours)):
                M = cv2.moments(contours[contour])
                try:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                except:
                    continue

                # img = cv2.circle(img, (center_x, center_y), 2, (255, 255, 255), -1)
                # img_center = cv2.circle(img_center, (center_x, center_y), center_rad, (cls_num), -1)
                img_center[center_y, center_x] = cls_num
                img_contour = cv2.drawContours(img_contour, contours, contour, (cls_num), contour_th)

        scio.savemat(os.path.join(save_dir, f'Label_cr_{center_rad}_ct_{contour_th}', save_name + ".mat"),
                     {'inst_map': inst_map,
                      'cls_map': cls_map,
                      'center_map': img_center,
                      'contour_map': img_contour})

        img_center[img_center != 0] = 255
        img_contour[img_contour != 0] = 255
        cv2.imwrite(os.path.join(save_dir, f'Centers_{center_rad}_{contour_th}', save_name + '.png'), img_center)
        cv2.imwrite(os.path.join(save_dir, f'Contours_{center_rad}_{contour_th}', save_name + '.png'), img_contour)