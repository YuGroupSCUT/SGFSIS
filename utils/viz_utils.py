import cv2
import math
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt


####
def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


####
def colorize(ch, vmin, vmax):
    """Will clamp value value outside the provided range to vmax and vmin."""
    cmap = plt.get_cmap("jet")
    ch = np.squeeze(ch.astype("float32"))
    vmin = vmin if vmin is not None else ch.min()
    vmax = vmax if vmax is not None else ch.max()
    ch[ch > vmax] = vmax  # clamp value
    ch[ch < vmin] = vmin
    ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    # take RGB from RGBA heat map
    ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    return ch_cmap


####
def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


####
def visualize_instances_map(
        input_image, inst_map, type_map=None, type_colour=None, line_thickness=2
):
    """Overlays segmentation results on image as contours.

    Args:
        input_image: input image
        inst_map: instance mask with unique value for every object
        type_map: type mask with unique value for every class
        type_colour: a dict of {type : colour} , `type` is from 0-N
                     and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours

    Returns:
        overlay: output image with segmentation overlay as contours
    """
    overlay = np.copy((input_image).astype(np.uint8))

    inst_list = list(np.unique(inst_map))  # get list of instances
    inst_list.remove(0)  # remove background

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object
        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # only has 1 instance per map, no need to check #contour detected by opencv
        contours_crop = np.squeeze(
            contours_crop[0][0].astype("int32")
        )  # * opencv protocol format may break
        contours_crop += np.asarray([[x1, y1]])  # index correction
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]
            type_id = np.unique(type_map_crop).max()  # non-zero
            inst_colour = type_colour[type_id]
        else:
            inst_colour = (inst_rng_colors[inst_idx]).tolist()
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)
    return overlay


####
def visualize_instances_dict(
        input_image, inst_dict, img_name, draw_dot=False, type_colour=None, line_thickness=2
):
    """Overlays segmentation results (dictionary) on image as contours.

    Args:
        input_image: input image
        inst_dict: dict of output prediction, defined as in this library
        draw_dot: to draw a dot for each centroid
        type_colour: a dict of {type_id : (type_name, colour)} ,
                     `type_id` is from 0-N and `colour` is a tuple of (R, G, B)
        line_thickness: line thickness of contours
    """
    overlay = np.copy((input_image))

    inst_rng_colors = np.array([255, 0, 0]).astype(np.uint8)

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = type_colour[inst_info["type"]][1]
        else:
            # inst_colour = (inst_rng_colors[idx]).tolist()
            inst_colour = (inst_rng_colors).tolist()
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, 1)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay


####
def gen_figure(
        imgs_list,
        titles,
        fig_inch,
        shape=None,
        share_ax="all",
        show=False,
        colormap=plt.get_cmap("jet"),
):
    """Generate figure."""
    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(
                axis="both",
                which="both",
                bottom="off",
                top="off",
                labelbottom="off",
                right="off",
                left="off",
                labelleft="off",
            )
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break

    fig.tight_layout()
    return fig

def draw_rand_inst_overlay_scaling(ori_img_, inst_map_, rand_color=True, draw_center=False):
    img = np.copy(ori_img_).astype(np.uint8)
    img = np.ascontiguousarray(img)
    # img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

    inst_map = np.copy(inst_map_)
    # inst_map = cv2.resize(inst_map, (512, 512), interpolation=cv2.INTER_NEAREST)

    insts = np.unique(inst_map).tolist()
    if 0 in insts:
        insts.remove(0)
    for ins in insts:
        inst_mask = np.copy(inst_map)
        inst_mask[inst_mask != ins] = 0


        inst_mask[inst_mask > 0] = 255
        inst_mask = inst_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in range(len(contours)):
            if draw_center:
                M = cv2.moments(contours[contour])
                try:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                except:
                    continue
                # img = cv2.circle(img, (center_x, center_y), 4, (255, 0, 0),  -1)
                img = cv2.circle(img, (center_x, center_y), 6, (255, 0, 0), -1)


            if rand_color:
                R = random.randint(0, 255)
                G = random.randint(0, 255)
                B = random.randint(0, 255)
                color = [R, G, B]
            else:
                color = [0, 255, 255]

            img = cv2.drawContours(img, contours, contour, color, 2, 8)

    # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    return img

def draw_overlay_scaling(ori_img_, inst_map_, cls_map_, col):
    img = np.copy(ori_img_).astype(np.uint8)
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    scale = 3

    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    inst_map = np.copy(inst_map_)
    inst_map = cv2.resize(inst_map, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    cls_map = np.copy(cls_map_)
    cls_map = cv2.resize(cls_map, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

    insts = np.unique(inst_map).tolist()
    if 0 in insts:
        insts.remove(0)
    for ins in insts:
        inst_mask = np.copy(inst_map)
        inst_mask[inst_mask != ins] = 0
        cls_num = list(set(cls_map[inst_mask == ins].flatten().tolist()))
        assert len(cls_num) == 1
        inst_mask[inst_mask > 0] = 255
        inst_mask = inst_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in range(len(contours)):
            try:
                # color = col[int(cls_num[0]) - 1].tolist()
                color = col[int(cls_num[0])].tolist()
            except:
                continue
            img = cv2.drawContours(img, contours, contour, color, 2, 8)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img

def draw_overlay(ori_img_, inst_map_, cls_map_, col):
    img = np.copy(ori_img_).astype(np.uint8)
    img = np.ascontiguousarray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    inst_map = np.copy(inst_map_)
    cls_map = np.copy(cls_map_)
    insts = np.unique(inst_map).tolist()
    insts.remove(0)
    for ins in insts:
        inst_mask = np.copy(inst_map)
        inst_mask[inst_mask != ins] = 0
        cls_num = list(set(cls_map[inst_mask == ins].flatten().tolist()))
        assert len(cls_num) == 1
        inst_mask[inst_mask > 0] = 255
        inst_mask = inst_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in range(len(contours)):
            try:
                color = col[int(cls_num[0])].tolist()
            except:
                continue
            img = cv2.drawContours(img, contours, contour, color, 1, 8)
    return img


def draw_overlay_by_index(ind_map, col):
    # bg = np.ones(shape=(ind_map.shape[0], ind_map.shape[1], 3)) * 255
    bg = np.zeros(shape=(ind_map.shape[0], ind_map.shape[1], 3))  # * 255
    bg = bg.astype(np.uint8)
    cls = np.unique(ind_map).tolist()
    if 0 in cls:
        cls.remove(0)
    for c in cls:
        # x = np.where(ind_map == c)
        bg[ind_map == c, :] = col[int(c)]
    return bg


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
