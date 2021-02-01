import copy
import os

import numpy as np
from pyquaternion import Quaternion

from data_classes import PointCloud, Box


def get_name_by_read_dir(path):
    indexs = []
    for every_file in os.listdir(path):
        index = every_file.split('.')
        if len(index) is 2:
            indexs.append(index)
    indexs.sort()
    count = 0
    str_count = ''
    if len(indexs) == 0:
        str_count = '000000'
        # print("None")
    else:
        for i in range(len(indexs)):
            count += 1
            if count < 10:
                str_count = '00000' + str(count)
            elif count < 100:
                str_count = '0000' + str(count)
            elif count < 1000:
                str_count = '000' + str(count)
            elif count < 10000:
                str_count = '00' + str(count)
            elif count < 100000:
                str_count = '0' + str(count)
            # print('str_count: ', str_count)
    return str_count


def bbox_label_save_txt(labels, path, name):
    label_file = os.path.join(path, name)
    fp = open(label_file, 'w')

    for j in range(labels.shape[0]):
        label = labels[j]
        if isinstance(label, str) is not True:
            label = str(label)
        if j < labels.shape[0] - 1:
            fp.write(label + " ")
        elif j == labels.shape[0] - 1:
            fp.write(label + "\r\n")
    fp.close()


def pc_save_pcd(points, path, name):
    # 存放路径
    PCD_FILE_PATH = os.path.join(path, name)
    if os.path.exists(PCD_FILE_PATH):
        os.remove(PCD_FILE_PATH)

    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部（重要）
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)
    handle.close()


def cropPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC


def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):
    new_PC = cropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC
