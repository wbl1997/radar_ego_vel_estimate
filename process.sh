# bagpath="/home/wbl/code/radar_lidar_imu_calib/data/data1_2024-07-04-13-13-10.bag"
# bagpath="/home/wbl/code/radar_lidar_imu_calib/data/data2_2024-07-05-10-23-47.bag"
# bagpath="/home/wbl/code/radar_lidar_imu_calib/data/0715/data1_2024-07-15-20-31-23.bag"
bagpath="/media/wbl/ZX2_NEW1/data/radar_egovel_est/0815/cfg3/data3_*.bag"

radar_topic="/ti_mmwave/radar_scan_pcl_0"
imu_topic="/imu/data"

# # lidar_imu_init
# source /home/wbl/code/radar_lidar_imu_calib/lidar_imu_ws/devel/setup.bash
# roslaunch lidar_imu_init livox_mid360_pcl.launch & sleep 2

# rosbag play $bagpath -r 3 -s 0

# wait

# # update fastlio extrinsic_R and extrinsic_T
# "/home/wbl/code/radar_lidar_imu_calib/script/updata_ext.sh" "mid360_dyn"

# fastlio: get pose_gt.txt
source /home/wbl/code/radar_lidar_imu_calib/lidar_imu_ws/devel/setup.bash
roslaunch fast_lio mapping_mid360_dyn.launch & sleep 2

rosbag play $bagpath -r 3 -s 0

# wait

# radar_imu_init (use vel): imu_T_radar
gt_vel="/home/wbl/code/radar_lidar_imu_calib/lidar_imu_ws/src/fastlio_mid360-master/Log/pos_log.txt"
calib_res_path="/home/wbl/code/radar_lidar_imu_calib/calib_result/"
# source activate radar_env
python3 /home/wbl/code/radar_lidar_imu_calib/script/vel_calib/vel_aided_radar_imu_calib_v3.py \
    $bagpath \
    $gt_vel \
    $calib_res_path \
    $radar_topic \
    $imu_topic

# # compute calibration: lidar_T_radar, imu_T_radar
# source activate radar_env
# calib_res_path="/home/wbl/code/radar_lidar_imu_calib/calib_result/"
# python /home/wbl/code/radar_lidar_imu_calib/script/compute_calib.py \
#     $calib_res_path
