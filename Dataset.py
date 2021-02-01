import os
import sys
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# ros
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append(ros_path)

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import *
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

if ros_path in sys.path:
    sys.path.remove(ros_path)

from dataset_utils import *
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

blue = lambda x: '\033[94m' + x + '\033[0m'
green = lambda x: '\033[1;32m' + x + '\033[0m'
yellow = lambda x: '\033[1;33m' + x + '\033[0m'
red = lambda x: '\033[1;31m' + x + '\033[0m'


class RosVisualizer():
    """
    1. 初始化ros节点，定义发布的话题类型，定义数据集路径
    2. 根据交互得到的scene的id，和类别，然后遍历所有，发布每一帧的点云以及box框信息，以及box中心点信息，均以ros话题形式发布

    注意：
        1. 数据集的路径dataset_path下应该类似以下目录格式
            .
            ├── calib
            ├── image_02
            ├── label_02
            └── velodyne

        2. 默认的话题发布频率需要手动设置，默认为10HZ
    """

    def __init__(self, args):
        # Init ros
        rospy.init_node('Ros_Visualizer', anonymous=True)
        self.predict_bbox = rospy.Publisher('/predict_bbox', MarkerArray, queue_size=1)
        self.point_pub = rospy.Publisher('/kitti_points', PointCloud2, queue_size=1)
        self.rate = rospy.Rate(args.pubRate)
        # Init Attributes
        self.KITTI_Folder = args.dataset_path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")
        self.replace = args.replace
        self.category = args.category
        self.save_path = args.save_path
        self.save_pcd = args.save_pcd
        print("KITTI_velo_path: ", self.KITTI_velo)
        print("KITTI_label_path: ", self.KITTI_label)

    def get_sceneID(self):
        try:
            # input_type = str(input("please input the scene split type[number/dataset]\n"))
            input_type = 'number'
            if input_type.upper() == 'number'.upper():
                print("valid scenes are: \n", os.listdir(self.KITTI_velo))
                scene = int(input("please input the scene number\n"))
                sceneID = scene
            elif input_type.upper() == 'dataset'.upper():
                scene = input("please input the dataset type['train'/'test'/'valid'/'all']\n")
                sceneID = self.getSceneList(scene)
            else:
                sceneID = None
                print(red("Input Error!!\n"), "please run again and input 'number' or 'dataset'\n")
                exit()
            print(yellow("sceneID is:"), sceneID)
            return sceneID
        except:
            print("something error! Exiting...")
            exit()

    def pub_pc_and_box(self, scene_id):
        pcd_path = os.path.join(self.save_path, self.category, 'lidar')
        label_path = os.path.join(self.save_path, self.category, 'label')

        self.make_sure_path_valid(pcd_path)
        self.make_sure_path_valid(label_path)

        if self.save_pcd and self.replace is True:
            shutil.rmtree(pcd_path)
            shutil.rmtree(label_path)
            os.mkdir(pcd_path)
            os.mkdir(label_path)

        scene_id = [scene_id] if isinstance(scene_id, int) else scene_id
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
               int(path) in scene_id
        ]

        print("list_of_scene: ", list_of_scene)

        # 遍历每一个序列列表中的序列
        for scene in list_of_scene:
            print("-" * 50)
            print("current scene id is: ", scene)
            # 标签路径
            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            # 读取标签txt文件
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df = df[df["type"] == self.category]  # 筛选出类别是car的标签
            df.insert(loc=0, column="scene", value=scene)  # 在标签中插入一列表示这是哪个场景
            # 还原索引，将df中的数据的每一行的索引变成默认排序的形式
            df = df.reset_index(drop=True)
            length = df.shape[0]
            frame_id = 0
            last_frame_id = 0
            corners_ = []

            try:
                for label_row in tqdm(range(length)):
                    this_anno = df.loc[label_row]
                    last_frame_id = frame_id
                    frame_id = this_anno['frame']
                    if frame_id != last_frame_id:
                        this_pc, this_box, state = self.getBBandPC(this_anno)  # this_pc's shape is (3, N)
                        if state is True:
                            points = this_pc.points.T
                            # pub box to show label
                            corners_.append(np.concatenate(this_box.corners().transpose(), axis=0))
                            self.vis_bbox(np.array(corners_))
                            corners_ = []
                            if self.save_pcd:
                                file_name = get_name_by_read_dir(pcd_path)
                                pc_save_pcd(points, pcd_path, file_name + '.pcd')
                            # pub whole frame pc
                            self.publish_pointcloud(points)
                            print("\n============================")
                            print(blue("scene: {} | frame: {}").format(this_anno['scene'], this_anno['frame']))
                            print(blue("pub pts with shape -> "), points.shape)

                            if cv2.waitKey(1) == 45:
                                exit()
                            self.rate.sleep()
                        else:
                            print(red("Error! getBBandPC error"))
                    else:
                        _, this_box, state = self.getBBandPC(this_anno)  # this_pc's shape is (3, N)
                        if state is True:
                            corners_.append(np.concatenate(this_box.corners().transpose(), axis=0).tolist())
                        else:
                            print(red("Error! getBBandPC error"))
            except KeyboardInterrupt:
                pass

    # 获取包含序列的列表
    def getSceneList(self, split):
        if "TRAIN" in split.upper():  # Training SET
            sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            sceneID = list(range(19, 21))
        else:  # Full Dataset
            sceneID = list(range(21))
        # logging.info("sceneID_path:\n%s\n", sceneID)   
        return sceneID

    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib', anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        # 在矩阵最下面叠加一行(0,0,0,1)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box, state = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box, state

    def getPCandBBfromPandas(self, box, calib):
        # 求出车辆的中心点 从此处的中心点是根据KITTI中相机坐标系下的中心点
        # 减去一半的高度移到地面上
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
            axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)  # 用中心点坐标和w,h,l以及旋转角来初始化BOX这个类
        State = True
        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         '%06d.bin' % (box["frame"]))  # f'{box["frame"]:06}.bin')
            # 从点云的.bin文件中读取点云数据并且转换为4*x的矩阵，且去掉最后的一行的点云的密度表示数据
            PC = PointCloud(np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            # 将点云转换到相机坐标系下　因为label中的坐标和h,w,l在相机坐标系下的
            PC.transform(calib)
        except FileNotFoundError:
            # logging.error("No such file found\n%s\n", velodyne_path)
            PC = PointCloud(np.array([[0, 0, 0]]).T)
            State = False

        return PC, BB, State

    def publish_pointcloud(self, pts):
        # pointcloud pub
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'velodyne'
        cloud_msg = pc2.create_cloud_xyz32(header, pts)
        self.point_pub.publish(cloud_msg)

    def vis_bbox(self, corners):
        all_bbox = MarkerArray()
        for i, corner in enumerate(corners):
            point_0 = Point(corner[0], corner[1], corner[2])
            point_1 = Point(corner[3], corner[4], corner[5])
            point_2 = Point(corner[6], corner[7], corner[8])
            point_3 = Point(corner[9], corner[10], corner[11])
            point_4 = Point(corner[12], corner[13], corner[14])
            point_5 = Point(corner[15], corner[16], corner[17])
            point_6 = Point(corner[18], corner[19], corner[20])
            point_7 = Point(corner[21], corner[22], corner[23])

            marker = Marker(id=i)
            marker.type = Marker.LINE_LIST
            marker.ns = 'velodyne'
            marker.action = Marker.ADD
            marker.header.frame_id = "/velodyne"
            marker.header.stamp = rospy.Time.now()

            marker.points.append(point_1)
            marker.points.append(point_2)
            marker.points.append(point_1)
            marker.points.append(point_0)
            marker.points.append(point_1)
            marker.points.append(point_5)
            marker.points.append(point_7)
            marker.points.append(point_4)
            marker.points.append(point_7)
            marker.points.append(point_6)
            marker.points.append(point_7)
            marker.points.append(point_3)
            marker.points.append(point_2)
            marker.points.append(point_6)
            marker.points.append(point_2)
            marker.points.append(point_3)
            marker.points.append(point_0)
            marker.points.append(point_4)
            marker.points.append(point_0)
            marker.points.append(point_3)
            marker.points.append(point_5)
            marker.points.append(point_6)
            marker.points.append(point_5)
            marker.points.append(point_4)

            marker.lifetime = rospy.Duration.from_sec(0.2)
            marker.scale.x = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.text = str(1)
            all_bbox.markers.append(marker)
        print(blue("pub markers with shape -> "), len(all_bbox.markers))
        self.predict_bbox.publish(all_bbox)

    @staticmethod
    def read_calib_file(filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        # 返回一个字典　字典中有6个键对　每个键对应的是calib文件中的一行，
        # key是'P0'，value是后面的对应的表示数值转换的一个3*4的numpy矩阵
        return data

    @staticmethod
    def make_sure_path_valid(dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate the pcd and label.txt file of fixed sequence in KITTI',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--category', required=False, type=str,
                        default='Car', help='category_name Car/Pedestrian/Van/Cyclist')
    parser.add_argument('--dataset_path', required=False, type=str,
                        default='/media/echo/仓库卷/DataSet/Autonomous_Driving/KITTI/tracking/origin_dataset/training',
                        help='dataset Path')
    parser.add_argument('--save_path', required=False, type=str,
                        default='saved',
                        help='save Path')
    parser.add_argument('--replace', required=False, type=bool,
                        default=True, help='whether delete the all files and generate again or not')
    parser.add_argument('--save_pcd', required=False, type=bool,
                        default=False, help='whether save whole frame pointcloud data as .pcd or not')
    parser.add_argument('--pubRate', required=False, type=int,
                        default=5, help='The rate of topic publish in ros. /Hz')

    args = parser.parse_args()
    kitti = RosVisualizer(args)
    scene_list = kitti.get_sceneID()
    kitti.pub_pc_and_box(scene_list)
