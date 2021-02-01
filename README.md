### 代码说明

本代码包使用python对kitti的`tracking`原始数据集做处理，首先索引目录下的序列，通过输入序列号[如果train的话是00-20]，来确定要可视化的序列，进而会将每一帧的点云以及点云中目标类别的label box通过ROS的可视化工具Rviz进行可视化，效果如下

![](1.png)

- ROS话题
  
  - `/predict_bbox`
  
    当前帧点云中对应类别的所有真值框，以marker的形式发出，方便在rviz上可视化
  
  - `/kitti_points`
  
    整帧点云话题，`frame_id`对应 `velodyne`，所有话题的发布频率都以激光雷达点云话题的频率为主，默认10HZ
  
  - `/box_centers`
  
    当前帧点云中对应类别的所有真值框的中心点，额外还加了帧号，格式为
  
    ```
    frame_num + " " + "box1_x" + " " + "box1_y" + " " + "box1_z" + " " + ...
    ```
  
  **注意：所有话题的stamp时间戳，都给成了帧号**
  
- 传入参数：
  
  - `category`
    
    - 要创建数据集的类别，KITTI中包含的类别有
    
      ```python
      'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
      ```
    
  - `dataset_path`
  
    - KITTI tracking原始数据集的路径，此路径下应包含
  
      ```python
      ├── calib
      ├── label_02
      └── velodyne
      	|--0000
          |--0001
      	|--...
      ```
  
  - `save_path`
  
    - 输出对应类别的数据集的路径，代码输出完成后，会在此目录下建立对应的类别文件夹，如下：
  
      ```python
      ├── your_category1
      │   ├── label
      │   ├── lidar
      ```
  
  - `replace`
  
    - 是否清空 `save_path`  下的文件重新生成？若选择否，则会计算原有目录下的文件数量，接着最后一个文件名的序号生成文件
    
  - `save_pcd`
  
    - 是否保存pcd格式的文件，是的话会将每一帧的整帧点云存为pcd
  
  - `pubRate`
  
    - 发布话题的频率，默认为10
  
- 环境要求：

  ```python
  pyquaternion
  pandas
  os
  tqdm
  argparse
  shutil  
  numpy
  ```
  
- 运行

  ```python
  python Dataset.py --category='Car' --dataset_path=<your_path> --save_path=<your_path> --replace=True --save_pcd=False --pubRate=10
  ```

  程序启动后会列出目录下的文件夹，并提示输入序列号

  ```python
  ❯ python Dataset.py
  KITTI_velo_path:  /media/echo/仓库卷/DataSet/Autonomous_Driving/KITTI/tracking/origin_dataset/training/velodyne
  KITTI_label_path:  /media/echo/仓库卷/DataSet/Autonomous_Driving/KITTI/tracking/origin_dataset/training/label_02
  valid seq are: 
   ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
  please input the scene number
  
  ```

  输入即可

