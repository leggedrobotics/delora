# Used Datasets

In general we provide support for rosbags and the kitti dataset. For each dataset we assume the following hierarchical
structure: ```dataset_name/<path_to_rosbag>``` or for KITTI its original sturcture ```dataset_name/sequence/scan```.
Here, sequences are numbered according to 00, 01, ...99. After prepocessing, scans will be numbered according to
00000...99999. An example for preprocessing a rosbag can be seen with the DARPA SubT dataset, the KITTI example can be
seen in the KITTI secion.

## Rosbag - DARPA SubT Dataset Example

Download the DARPA SubT Rosbags: [link](https://bitbucket.org/subtchallenge/subt_reference_datasets/src/master/)

```bash
mkdir $PWD/datasets/darpa/
# Link taken from https://bitbucket.org/subtchallenge/subt_reference_datasets/src/master/
wget https://subt-data.s3.amazonaws.com/SubT_Urban_Ckt/a_lvl_1.bag -O $PWD/datasets/darpa/00.bag

```

### Structure

### Run preprocessing

Pull the rosbag at the above link, and put it to ```<delora_ws>/datasets/darpa/<name>.bag```. Rename it
to ```<delora_ws>/datasets/darpa/00.bag``` (or ```01...99.bag``` if you have multiple sequences). In the
file ```./config/deployment_options.yaml``` set ```datasets: ["darpa"]```. Preprocessing can then be run with the
following command:

```bash
preprocess_data.py
```

If your files are placed somewhere else, simply adapt the path in ```./config/config_datasets.yaml``` (global or local
w.r.t. to python working directory).

## KITTI Dataset
### LiDAR Scans
Download the "velodyne laster data" from the official KITTI odometry evaluation (
80GB): [link](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Put it to ```<delora_ws>/datasets/kitti```,
where ```kitti``` contains ```/data_odometry_velodyne/dataset/sequences/00..21```.
### Groundtruth poses
Please also download the groundtruth poses [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
Make sure that the files are located at ```<delora_ws>/datasets/kitti```,
where ```kitti``` contains ```/data_odometry_poses/dataset/poses/00..10.txt```.

### Run preprocessing

In the file ```./config/deployment_options.yaml``` set ```datasets: ["kitti"]```. Then run

```bash
preprocess_data.py
```

## Custom Dataset

Just follow the above procedure for custom datasets. Any sequence of rosbags can be used.

## Visualize Processed Dataset

The point cloud and its estimated normals for a dataset can be visualized using the following command:

```bash
visualize_pointcloud_normals.py
```

With this command, the first 100 scans with its normals are published under the topics ```/lidar/points```
and ```/lidar/normals``` in the frame ```lidar``` and can be visualized in *RVIZ*.
