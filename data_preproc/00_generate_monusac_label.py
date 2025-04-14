#Process whole slide images
import os
import numpy as np
import openslide
from glob import glob
# import cv2
import tqdm
from shapely.geometry import Polygon
from skimage import draw
import xml.etree.ElementTree as ET
import scipy.io as sio
import random

type_dict = {'Ambiguous': 5,
             'Epithelial': 1,
             'Lymphocyte': 2,
             'Macrophage': 3,
             'Neutrophil': 4}

# Read svs files from the desired path
count = 0
data_path = "/home/data1/my/dataset/monusac/MoNuSAC_images_and_annotations/"  # Path to read data from


# os.chdir(destination_path + 'MoNuSAC_masks')  # Create folder named as MoNuSAC_masks
patients = [x[0] for x in os.walk(data_path)]  # Total patients in the data_path
random.seed(2023)
random.shuffle(patients)
train_patients = patients[:int(len(patients) * 0.85)]
valid_patients = patients[int(len(patients) * 0.85):]

fold = "Train"
for patient_loc in tqdm.tqdm(train_patients):
    patient_name = patient_loc[len(data_path) + 1:]  # Patient name

    ## Read sub-images of each patient in the data path
    sub_images = glob(patient_loc + '/*.svs')
    for sub_image_loc in sub_images:
        gt = 0
        sub_image_name = os.path.basename(sub_image_loc).split('.')[0]
        # sub_image = os.path.join(destination_path + 'MoNuSAC_masks', patient_name,
        #                          sub_image_name)  # './' + patient_name + '/' + sub_image_name  # Destination path

        image_name = sub_image_loc
        img = openslide.OpenSlide(image_name)
        # x = np.array(img.read_region((0, 0), 0, img.level_dimensions[0]).convert('RGB'))

        # set inst and type_map
        inst_map = np.transpose(np.zeros((img.read_region((0, 0), 0, img.level_dimensions[0]).size)))
        type_map = np.transpose(np.zeros((img.read_region((0, 0), 0, img.level_dimensions[0]).size)))

        # If svs image needs to save in tif
        img_2_save = img.read_region((0, 0), 0, img.level_dimensions[0]).convert('RGB')
        img_2_save.save(os.path.join(f'/home/data1/my/dataset/MoNuSAC/{fold}/Images/', sub_image_name + '.png'))


        # Read xml file
        xml_file_name = image_name[:-4]
        xml_file_name = xml_file_name + '.xml'
        tree = ET.parse(xml_file_name)
        root = tree.getroot()

        # Generate binary mask for each cell-type
        for k in range(len(root)):
            label = [x.attrib['Name'] for x in root[k][0]]
            label = label[0]

            for child in root[k]:
                for x in child:
                    r = x.tag
                    # if r == 'Attribute':
                    #     count = count + 1
                    #     label = x.attrib['Name']
                    #     sub_path = sub_image + '/' + label
                    #
                    #     try:
                    #         os.mkdir(sub_path)
                    #     except OSError:
                    #         pass
                    #     else:
                    #         pass

                    if r == 'Region':
                        regions = []
                        vertices = x[1]
                        coords = np.zeros((len(vertices), 2))
                        for i, vertex in enumerate(vertices):
                            coords[i][0] = vertex.attrib['X']
                            coords[i][1] = vertex.attrib['Y']
                        regions.append(coords)
                        try:
                            poly = Polygon(regions[0])
                        except:
                            os.remove(os.path.join(f'/home/data1/my/dataset/MoNuSAC/{fold}/Images/', sub_image_name + '.png'))
                            continue

                        vertex_row_coords = regions[0][:, 0]
                        vertex_col_coords = regions[0][:, 1]
                        fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, inst_map.shape)
                        type_map[fill_row_coords, fill_col_coords] = type_dict[label]
                        inst_map[fill_row_coords, fill_col_coords] = gt
                        gt = gt + 1  # Keep track of giving unique valu to each instance in an image
                        label_save = {'inst_map': inst_map, 'type_map': type_map}
                        sio.savemat(os.path.join(f'/home/data1/my/dataset/MoNuSAC/{fold}/Labels/', sub_image_name + '.mat'), label_save)

fold = 'Valid'
for patient_loc in tqdm.tqdm(valid_patients):
    patient_name = patient_loc[len(data_path) + 1:]  # Patient name

    ## Read sub-images of each patient in the data path
    sub_images = glob(patient_loc + '/*.svs')
    for sub_image_loc in sub_images:
        gt = 0
        sub_image_name = os.path.basename(sub_image_loc).split('.')[0]
        # sub_image = os.path.join(destination_path + 'MoNuSAC_masks', patient_name,
        #                          sub_image_name)  # './' + patient_name + '/' + sub_image_name  # Destination path

        image_name = sub_image_loc
        img = openslide.OpenSlide(image_name)
        # x = np.array(img.read_region((0, 0), 0, img.level_dimensions[0]).convert('RGB'))

        # set inst and type_map
        inst_map = np.transpose(np.zeros((img.read_region((0, 0), 0, img.level_dimensions[0]).size)))
        type_map = np.transpose(np.zeros((img.read_region((0, 0), 0, img.level_dimensions[0]).size)))

        # If svs image needs to save in tif
        img_2_save = img.read_region((0, 0), 0, img.level_dimensions[0]).convert('RGB')
        img_2_save.save(os.path.join(f'/home/data1/my/dataset/MoNuSAC/{fold}/Images/', sub_image_name + '.png'))

        # Read xml file
        xml_file_name = image_name[:-4]
        xml_file_name = xml_file_name + '.xml'
        tree = ET.parse(xml_file_name)
        root = tree.getroot()

        # Generate binary mask for each cell-type
        for k in range(len(root)):
            label = [x.attrib['Name'] for x in root[k][0]]
            label = label[0]

            for child in root[k]:
                for x in child:
                    r = x.tag
                    # if r == 'Attribute':
                    #     count = count + 1
                    #     label = x.attrib['Name']
                    #     sub_path = sub_image + '/' + label
                    #
                    #     try:
                    #         os.mkdir(sub_path)
                    #     except OSError:
                    #         pass
                    #     else:
                    #         pass

                    if r == 'Region':
                        regions = []
                        vertices = x[1]
                        coords = np.zeros((len(vertices), 2))
                        for i, vertex in enumerate(vertices):
                            coords[i][0] = vertex.attrib['X']
                            coords[i][1] = vertex.attrib['Y']
                        regions.append(coords)
                        try:
                            poly = Polygon(regions[0])
                        except:
                            os.remove(
                                os.path.join(f'/home/data1/my/dataset/MoNuSAC/{fold}/Images/', sub_image_name + '.png'))
                            continue

                        vertex_row_coords = regions[0][:, 0]
                        vertex_col_coords = regions[0][:, 1]
                        fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords,
                                                                        inst_map.shape)
                        type_map[fill_row_coords, fill_col_coords] = type_dict[label]
                        inst_map[fill_row_coords, fill_col_coords] = gt
                        gt = gt + 1  # Keep track of giving unique valu to each instance in an image
                        label_save = {'inst_map': inst_map, 'type_map': type_map}
                        sio.savemat(os.path.join(f'/home/data1/my/dataset/MoNuSAC/{fold}/Labels/', sub_image_name + '.mat'),
                                    label_save)
