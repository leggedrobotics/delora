# DeLORA: Self-supervised Deep LiDAR Odometry for Robotic Applications

![title_img](images/title_img.png)

**The work is currently under review. The corresponding code will be published upon publication of the paper.**
From the Robotics Systems Lab at ETH Zurich, Switzerland.

# What will be released
We will release the full code to train the proposed model and to do inference on the dataset. Additionally, to enable the community to test the model in ROS, we will also provide a corresponding ROS-node which deploys the model for inference. For this, a pre-trained model for the KITTI dataset will be made publicly available as well.
Furthermore, we will release the code for the conversion of the KITTI (http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and the DARPA SubT (https://bitbucket.org/subtchallenge/subt_reference_datasets/src/master/) datasets to the used format, as well as code for computing the normal vectors as a pre-processing step.

# Paper Abstract
Reliable robot pose estimation is a key building block of many robot autonomy pipelines, with LiDAR localization being an active research domain. In this work, a versatile self-supervised LiDAR odometry estimation method is presented, in order to enable the efficient utilization of all available LiDAR data while maintaining real-time performance. The proposed approach selectively applies geometric losses during training, being cognizant of the amount of information that can be extracted from scan points. In addition, no labeled or ground-truth data is required, hence making the presented approach suitable for pose estimation in applications where accurate ground-truth is difficult to obtain. Furthermore, the presented network architecture is applicable to a wide range of environments and sensor modalities without requiring any network or loss function adjustments. The proposed approach is thoroughly tested for both indoor and outdoor real-world applications through a variety of experiments using legged, tracked and wheeled robots, demonstrating the suitability of learning-based LiDAR odometry for complex robotic applications.

# Resources
Paper: https://arxiv.org/pdf/2011.05418.pdf

Video: https://youtu.be/rgeRseUTryA
