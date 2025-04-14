import os
import cv2
import numpy as np
from skimage import morphology
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import warnings


def noop(*args, **kargs):
    pass


warnings.warn = noop


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


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def water_shed_with_marker(img, contour, centroid):
    img_post = np.copy(img)
    img_contour = np.copy(contour)

    img_post = binary_fill_holes(img_post).astype("uint8")
    img_maker = np.copy(img_post)

    # contour guidance
    distance = ndimage.morphology.distance_transform_edt(img_post)
    img_maker[img_contour != 0] = 0
    marker = img_maker

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = ndimage.label(marker)[0]

    ################################## centroid control #############################
    # if isinstance(cen_guid_map, np.ndarray):
    marker_with_points_control = np.copy(marker)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cen_guid_map = cv2.dilate(centroid.astype(np.float), kernel, iterations=1)
    cen_guid_map = ndimage.label(cen_guid_map)[0]
    marker_inst = np.unique(marker).tolist()
    if 0 in marker_inst:
        marker_inst.remove(0)
    for inst in marker_inst:
        cen_guid_map_cp = np.copy(cen_guid_map)
        cen_guid_map_cp[marker != inst] = 0
        point_marker = np.unique(cen_guid_map_cp).tolist()
        # if more than 2 point markers (exclude background) in a same marker
        if len(point_marker) > 2:
            ################# implement a wotersed for the marker map ##############
            region_marker = np.zeros_like(marker)
            region_marker[marker == inst] = 1
            marker_distance = ndimage.morphology.distance_transform_edt(region_marker)
            # remove the original marker
            marker_with_points_control[marker == inst] = 0
            # split new marker
            region_marker = watershed(-marker_distance, cen_guid_map_cp, mask=region_marker)
            # add to original marker map
            marker_with_points_control += region_marker

    marker_with_points_control[marker_with_points_control != 0] = 1
    marker = ndimage.label(marker_with_points_control)[0]
    #################################################################################

    img_post = watershed(-distance, marker, mask=img_post)

    return img_post

####
def __proc_np_cp(pred, sem_guid_map=None, cen_guid_map=None, con_guid_map=None,):
    seg_output = pred[..., 0]
    contour_output = pred[..., 1]
    centroid_output = pred[..., 2]
    seg_output = np.array(seg_output >= 0.5, dtype=np.int32)
    contour_output = np.array(contour_output >= 0.5, dtype=np.int32)
    centroid_output = np.array(centroid_output >= 0.5, dtype=np.int32)

    if isinstance(sem_guid_map, np.ndarray):
        sem_guid_map = np.array(sem_guid_map >= 0.5, dtype=np.int32)
        seg_output[sem_guid_map != 0] = 1
    if isinstance(con_guid_map, np.ndarray):
        con_guid_map = np.array(con_guid_map >= 0.5, dtype=np.int32)
        contour_output[con_guid_map != 0] = 1
    if isinstance(cen_guid_map, np.ndarray):
        cen_guid_map = np.array(cen_guid_map >= 0.5, dtype=np.int32)
        centroid_output[cen_guid_map != 0] = 1

    query_inst_proc = water_shed_with_marker(seg_output, contour_output, centroid_output)
    proced_pred = remove_small_objects(query_inst_proc, min_size=16)

    return proced_pred


def process(pred_map, sem_guid_map=None, cen_guid_map=None, con_guid_map=None,
            nr_types=None, return_centroids=False):

    """Post processing script for image tiles.
    """
    if nr_types is not None:
        pred_type = pred_map[..., -1]
        pred_inst = pred_map[..., :3]
        pred_type = pred_type.astype(np.int32)
    else:
        pred_inst = pred_map

    pred_inst = np.squeeze(pred_inst)
    pred_inst = __proc_np_cp(pred_inst,
                             sem_guid_map=sem_guid_map,
                             cen_guid_map=cen_guid_map,
                             con_guid_map=con_guid_map)

    inst_info_dict = None
    if return_centroids or nr_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                       inst_bbox[0][0]: inst_bbox[1][0], inst_bbox[0][1]: inst_bbox[1][1]
                       ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                    inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # print('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    return pred_inst, inst_info_dict

