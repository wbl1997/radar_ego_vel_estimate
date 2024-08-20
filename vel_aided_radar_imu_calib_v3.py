import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from scipy.linalg import solve, lstsq
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import open3d as o3d

save_path = "/home/wbl/code/visual_radar_slam/project_radar2img/result/pc_img/"

def rotation_matrix_from_vector(vec):
    return R.from_rotvec(vec).as_matrix()

def align_error(params, P, Q, W, weights):
    # Extract rotation vector and convert to matrix
    rot_vec = params[:3]
    R_matrix = rotation_matrix_from_vector(rot_vec)
    
    # Extract lever arm
    lever_arm = params[3:]
    
    # Apply rotation and lever arm transformation
    # P_transformed = P @ R_matrix.T + np.cross(W, lever_arm)
    # P_transformed = (R_matrix @ P.T - np.cross(W, lever_arm).T).T
    Q_transformed = (Q + np.cross(W, lever_arm)) @ R_matrix
    
    # Compute weighted error
    # residuals = np.linalg.norm((P_transformed - Q), axis=1)
    residuals = np.linalg.norm((P - Q_transformed), axis=1)
    nan_mask = np.isnan(residuals)
    for i in range(len(residuals)):
        if nan_mask[i]:
            residuals[i] = 0
    has_nan = np.any(residuals)
    weighted_error = np.sum(weights * residuals ** 2)
    
    return weighted_error

def compute_weights(P, Q, R_matrix, W, lever_arm, k=1.0):
    # Transform P by R_matrix and lever arm
    # P_transformed = P @ R_matrix.T + np.cross(W, lever_arm)
    # P_transformed = (R_matrix @ P.T - np.cross(W, lever_arm).T).T
    Q_transformed = (Q + np.cross(W, lever_arm)) @ R_matrix
    # Compute residuals
    # residuals = np.linalg.norm(P_transformed - Q, axis=1)
    residuals = np.linalg.norm(P - Q_transformed, axis=1)
    nan_mask = np.isnan(residuals)
    for i in range(len(residuals)):
        if nan_mask[i]:
            residuals[i] = 100000
    # Compute weights
    weights = 1 / (residuals + k)
    return weights

def kabsch_algorithm_irls(P, Q, W, max_iters=100, k=1.0):
    # Initialize rotation parameters (rotation vector) and lever arm
    params = np.zeros(6)
    
    # Initialize weights
    weights = np.ones(P.shape[0])
    
    for i in range(max_iters):
        # Optimize rotation parameters and lever arm
        result = minimize(align_error, params, args=(P, Q, W, weights), method='BFGS')
        params = result.x
        
        # Compute rotation matrix
        R_matrix = rotation_matrix_from_vector(params[:3])
        lever_arm = params[3:]
        
        # Update weights
        weights = compute_weights(P, Q, R_matrix, W, lever_arm, k)
    
    R_matrix = rotation_matrix_from_vector(params[:3])
    lever_arm = params[3:]
    return R_matrix, lever_arm

def read_imu_data(bag_file, topic):
    bag = rosbag.Bag(bag_file)
    timestamps = []
    angular_velocities = []

    for topic, msg, t in bag.read_messages(topics=[topic]):
        timestamps.append(t.to_sec())
        angular_velocity = msg.angular_velocity
        angular_velocities.append([angular_velocity.x, angular_velocity.y, angular_velocity.z])

    bag.close()
    return np.array(timestamps), np.array(angular_velocities)

def read_tum_poses(file_path):
    data = np.loadtxt(file_path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    orientations = data[:, 4:8]
    return timestamps, positions, orientations

def read_tum_vel(file_path):
    data = np.loadtxt(file_path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    orientations = data[:, 4:8]
    velocities = data[:, 8:11]
    return timestamps, velocities

def compute_velocities(timestamps, positions):
    dt = np.diff(timestamps)
    velocities = np.diff(positions, axis=0) / dt[:, None]
    return velocities, timestamps[1:]

def compute_and_transform_velocities(timestamps, positions, orientations=None):
    dt = np.diff(timestamps)
    velocities = np.diff(positions, axis=0) / dt[:, None]

    if orientations is not None:
        body_velocities = []
        for v, q in zip(velocities, orientations[:-1]):
            r = R.from_quat(q)
            body_vel = r.inv().apply(v)
            body_velocities.append(body_vel)
        return np.array(body_velocities), timestamps[1:]
    else:
        return velocities, timestamps[1:]

def read_bag_data(bag_name, topic, field_names=("x", "y", "z", "velocity", "intensity")):
    bag = rosbag.Bag(bag_name)
    radar_data = []
    doppler_data = []
    timestamps = []
    for topic, msg, t in bag.read_messages(topics=[topic]):
        points = pc2.read_points(msg, field_names=field_names, skip_nans=True)
        radar_scan = []
        doppler_scan = []
        for point in points:
            if field_names[3] == 'velocity' or field_names[3] == 'doppler':
                radar_scan.append([point[0], point[1], point[2], point[4]])
                doppler_scan.append(point[3])
            elif field_names[4] == 'doppler' or field_names[4] == 'velocity':
                radar_scan.append([point[0], point[1], point[2], point[3]])
                doppler_scan.append(point[4])
        radar_data.append(np.array(radar_scan))
        doppler_data.append(np.array(doppler_scan))
        timestamps.append(t.to_sec())
    bag.close()
    return radar_data, doppler_data, timestamps

def radar_ego_velocity_estimate(radar_scan, v_dopplers, config):
    valid_targets = []
    for i, target in enumerate(radar_scan):
        r = np.linalg.norm(target[:3])
        azimuth = np.arctan2(target[1], target[0])
        elevation = np.arctan2(target[2], np.sqrt(target[0]**2 + target[1]**2))
        v_d = v_dopplers[i]

        if config['min_dist'] < r < config['max_dist'] and \
           np.abs(azimuth) < np.radians(config['azimuth_thresh_deg']) and \
           np.abs(elevation) < np.radians(config['elevation_thresh_deg']):
            valid_targets.append(np.hstack((azimuth, elevation, target[:3], target[:3]/r, -v_d, target[3])))
    print("len valid: ", len(valid_targets))
    if len(valid_targets) > 5:
        radar_data = np.array(valid_targets)[:, [5, 6, 7, 8]]
        success, v_r, P_v_r, inlier_idx_best = solve_3d_lsq_irls(radar_data, config)
        if success:
            radar_scan_inlier = [valid_targets[idx][2:5] for idx in inlier_idx_best]
            radar_scan_outlier = [valid_targets[idx][2:5] for idx in range(len(valid_targets)) if idx not in inlier_idx_best]
        else:
            radar_scan_inlier = None
            radar_scan_outlier = None
        return success, v_r, P_v_r, radar_scan_inlier, radar_scan_outlier
    else:
        return False, None, None, None, None

def solve_3d_lsq_irls(radar_data, config):
    H = radar_data[:, :3]
    y = radar_data[:, 3]
    
    p = 1
    e = np.ones(radar_data.shape[0])
    wgt_vec = np.ones(radar_data.shape[0]) / radar_data.shape[0]
    Wgt_mat = np.eye(radar_data.shape[0]) / radar_data.shape[0]
    err_pre = 100000
    sigma_v_r = np.zeros(3)
    inlier_idx = []

    for k in range(config['irls_iter']):
        e = np.abs(e)
        e = np.maximum(e, 1e-15)

        wgt_vec = np.power(e, (p-2)/2)
        Wgt_mat = np.diag(wgt_vec / np.sum(wgt_vec))

        HTH = H.T @ Wgt_mat.T @ Wgt_mat @ H
        HTy = H.T @ Wgt_mat.T @ Wgt_mat @ y

        if config['use_cholesky_instead_of_bdcsvd']:
            v_r = solve(HTH, HTy)
        else:
            v_r, _, _, _ = lstsq(Wgt_mat @ H, Wgt_mat @ y)

        e = H @ v_r - y
        P_v_r = (Wgt_mat @ e).T @ (Wgt_mat @ e) * np.linalg.inv(HTH) / (H.shape[0] - 3)
        sigma_v_r = np.diag(P_v_r)

        inlier_idx = [j for j in range(e.shape[0]) if np.abs(e[j]) < config['inlier_thresh']]

        if k > 0 and np.abs(err_pre - ((Wgt_mat @ e).T @ (Wgt_mat @ e))) < 1e-8:
            break
        err_pre = (Wgt_mat @ e).T @ (Wgt_mat @ e)

    inlier_idx_best = inlier_idx

    N_in = len(inlier_idx_best)
    HTH = (N_in * Wgt_mat @ H).T @ (N_in * Wgt_mat @ H)
    P_v_r = (N_in * Wgt_mat @ e).T @ (N_in * Wgt_mat @ e) * np.linalg.inv(HTH) / (H.shape[0] - 3)
    sigma_v_r = np.diag(P_v_r)
    offset = np.array([config['sigma_offset_radar_x'], config['sigma_offset_radar_y'], config['sigma_offset_radar_z']])**2
    P_v_r += np.diag(offset)

    if np.all(sigma_v_r >= 0) and \
       sigma_v_r[0] < config['max_sigma_x'] and \
       sigma_v_r[1] < config['max_sigma_y'] and \
       sigma_v_r[2] < config['max_sigma_z']:
        return True, v_r, P_v_r, inlier_idx_best
    else:
        return False, None, None, None

def align_velocities(v_radar, v_imu, w_imu):
    R, b_l_r = kabsch_algorithm_irls(v_radar, v_imu, w_imu)
    v_radar_aligned = v_radar @ R.T + np.cross(w_imu, b_l_r)
    return v_radar_aligned, R, b_l_r

def test_radar_velocity_estimation(bag_name, radar_topic, imu_topic, gt_folder, config, field_names=("x", "y", "z", "doppler", "intensity")):
    radar_data, doppler_data, timestamps = read_bag_data(bag_name, radar_topic, field_names)
    imu_timestamps, angular_velocities = read_imu_data(bag_name, imu_topic)

    ts_gt, pos_gt, ori_gt = read_tum_poses(gt_folder)
    gt_vel1, gt_ts1 = compute_and_transform_velocities(ts_gt, pos_gt, ori_gt)

    estimated_velocities = []
    config['angle_wgt'] = False
    for scan, dopplers in zip(radar_data, doppler_data):
        success, v_r, P_v_r, inlier, outlier = radar_ego_velocity_estimate(scan, dopplers, config)
        if success:
            estimated_velocities.append(v_r)
            print('Estimated velocity: ', v_r)
        else:
            estimated_velocities.append([np.nan, np.nan, np.nan])
    estimated_velocities_angless = np.array(estimated_velocities)
    estimated_velocities_ang = estimated_velocities_angless

    # estimated_velocities = []
    # config['angle_wgt'] = True
    # for scan, dopplers in zip(radar_data, doppler_data):
    #     success, v_r, P_v_r, inlier, outlier = radar_ego_velocity_estimate(scan, dopplers, config)
    #     if success:
    #         estimated_velocities.append(v_r)
    #         print('Estimated velocity: ', v_r)
    #     else:
    #         estimated_velocities.append([np.nan, np.nan, np.nan])
    # estimated_velocities_ang = np.array(estimated_velocities)

    plt.figure()
    plt.plot(timestamps, np.linalg.norm(estimated_velocities_angless, axis=1), label='Estimated Norm Velocity w/o ang wgt')
    # plt.plot(timestamps, np.linalg.norm(estimated_velocities_ang, axis=1), label='Estimated Norm Velocity w/ ang wgt')

    plt.plot(gt_ts1, np.linalg.norm(gt_vel1, axis=1), label='Ground Truth Norm Velocity', linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.show()

    common_start_time = max(timestamps[0], gt_ts1[0])
    common_end_time = min(timestamps[-1], gt_ts1[-1])
    common_timestamps = np.arange(common_start_time, common_end_time, 0.1)  # 10Hz

    interp_estimated_velocities_angless = np.array([
        np.interp(common_timestamps, timestamps, estimated_velocities_angless[:, i])
        for i in range(3)
    ]).T
    interp_estimated_velocities_ang = np.array([
        np.interp(common_timestamps, timestamps, estimated_velocities_ang[:, i])
        for i in range(3)
    ]).T
    interp_gt_velocities = np.array([
        np.interp(common_timestamps, gt_ts1, gt_vel1[:, i])
        for i in range(3)
    ]).T

    # # save interpolated data as txt
    # np.savetxt('/home/wbl/code/visual_radar_slam/project_radar2img/result/vel_calib/interp_estimated_velocities_angless.txt', interp_estimated_velocities_angless)
    # np.savetxt('/home/wbl/code/visual_radar_slam/project_radar2img/result/vel_calib/interp_estimated_velocities_ang.txt', interp_estimated_velocities_ang)
    # np.savetxt('/home/wbl/code/visual_radar_slam/project_radar2img/result/vel_calib/interp_gt_velocities.txt', interp_gt_velocities)
    
    return interp_estimated_velocities_angless, interp_estimated_velocities_ang, interp_gt_velocities, common_timestamps

# Example configuration dictionary
# config = {
#     'irls_iter': 100,
#     'use_cholesky_instead_of_bdcsvd': True,
#     'inlier_thresh': 0.15,
#     'min_dist': 1.0,
#     'max_dist': 500.0,
#     'azimuth_thresh_deg': 360, #60,
#     'elevation_thresh_deg': 360, #40,
#     'allowed_outlier_percentage': 0.25,
#     'thresh_zero_velocity': 0.05,
#     'sigma_zero_velocity_x': 0.025,
#     'sigma_zero_velocity_y': 0.025,
#     'sigma_zero_velocity_z': 0.025,
#     'sigma_offset_radar_x': 0.025,
#     'sigma_offset_radar_y': 0.025,
#     'sigma_offset_radar_z': 0.025,
#     'max_sigma_x': 0.2,
#     'max_sigma_y': 0.2,
#     'max_sigma_z': 0.2,
#     'use_ransac': True,
#     'angle_wgt': True
# }

# config = {
#     'irls_iter': 100,
#     'use_cholesky_instead_of_bdcsvd': True,
#     'inlier_thresh': 0.35,
#     'min_dist': 1.0,
#     'max_dist': 100.0,
#     'azimuth_thresh_deg': 180, #60,
#     'elevation_thresh_deg': 180, #40,
#     'allowed_outlier_percentage': 0.40,
#     'thresh_zero_velocity': 0.1,
#     'sigma_zero_velocity_x': 0.025,
#     'sigma_zero_velocity_y': 0.025,
#     'sigma_zero_velocity_z': 0.025,
#     'sigma_offset_radar_x': 0.025,
#     'sigma_offset_radar_y': 0.025,
#     'sigma_offset_radar_z': 0.025,
#     'max_sigma_x': 0.2,
#     'max_sigma_y': 0.2,
#     'max_sigma_z': 0.2,
#     'use_ransac': True,
#     'angle_wgt': True
# }

config = {
    'irls_iter': 100,
    'use_cholesky_instead_of_bdcsvd': True,
    'inlier_thresh': 0.15,
    'min_dist': 0.5,
    'max_dist': 30.0,
    'azimuth_thresh_deg': 60,
    'elevation_thresh_deg': 20,
    'allowed_outlier_percentage': 0.25,
    'thresh_zero_velocity': 0.1,
    'sigma_zero_velocity_x': 0.025,
    'sigma_zero_velocity_y': 0.025,
    'sigma_zero_velocity_z': 0.025,
    'sigma_offset_radar_x': 0.025,
    'sigma_offset_radar_y': 0.025,
    'sigma_offset_radar_z': 0.025,
    'max_sigma_x': 0.2,
    'max_sigma_y': 0.2,
    'max_sigma_z': 0.2,
    'use_ransac': True,
    'angle_wgt': True
}


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python radar_ego_velocity_estimate.py path_to_bag.bag imu_topic gt_folder')
        sys.exit(1)
    bag_path = sys.argv[1]
    gt_folder = sys.argv[2]
    save_transfrom_path = sys.argv[3]
    if not os.path.exists(save_transfrom_path):
        print('Save rotation matrix path does not exist')
    save_transfrom_path += "imu_T_radar.txt"
    radar_topic = '/ti_mmwave/radar_scan_pcl_0'
    imu_topic = '/imu/data'
    if len(sys.argv) > 3:
        radar_topic = sys.argv[4]
        imu_topic = sys.argv[5]
    
    if radar_topic=="/ars548":
        field_names = ("x", "y", "z", "doppler", "intensity")
    else:
        field_names = ("x", "y", "z", "intensity", "velocity")

    vel_angless, vel_ang, vel_gt, common_timestamps = test_radar_velocity_estimation(bag_path, radar_topic, imu_topic, gt_folder, config, field_names=field_names)
    
    # Get corresponding IMU angular velocities
    imu_timestamps, angular_velocities = read_imu_data(bag_path, imu_topic)
    common_w = np.array([
        np.interp(common_timestamps, imu_timestamps, angular_velocities[:, i])
        for i in range(3)
    ]).T
    
    aligned_vel_angless, rotation_matrix, lever_arm = align_velocities(vel_angless, vel_gt, common_w)
    
    print("Rotation matrix:\n", rotation_matrix)
    print("Lever arm:\n", lever_arm)
    
    # aligned_vel_angless = vel_angless @ rotation_matrix.T + np.cross(common_w, lever_arm)
    aligned_vel_angless = (rotation_matrix @ vel_angless.T - np.cross(common_w, lever_arm).T).T
    err = aligned_vel_angless - vel_gt
    # nan found and remove
    nan_mask = np.isnan(err)
    for i in range(len(err)):
        if nan_mask[i].any():
            err[i] = 0
    mean_error = np.mean(np.linalg.norm(err, axis=1))
    print("Mean error: ", mean_error)
    
    # Save rotation matrix and lever arm
    transfrom_matrix = np.eye(4)
    transfrom_matrix[:3, :3] = rotation_matrix
    transfrom_matrix[:3, 3] = lever_arm
    np.savetxt(save_transfrom_path, transfrom_matrix, fmt='%.6f')

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(vel_angless[:, 0], label='Estimated Velocity X w/o ang wgt')
    axes[0].plot(aligned_vel_angless[:, 0], label='Aligned Estimated Velocity X w/o ang wgt')
    axes[0].plot(vel_gt[:, 0], label='Ground Truth Velocity X', linestyle='dashed')
    axes[0].set_ylabel('Velocity X (m/s)')
    axes[0].legend()

    axes[1].plot(vel_angless[:, 1], label='Estimated Velocity Y w/o ang wgt')
    axes[1].plot(aligned_vel_angless[:, 1], label='Aligned Estimated Velocity Y w/o ang wgt')
    axes[1].plot(vel_gt[:, 1], label='Ground Truth Velocity Y', linestyle='dashed')
    axes[1].set_ylabel('Velocity Y (m/s)')
    axes[1].legend()

    axes[2].plot(vel_angless[:, 2], label='Estimated Velocity Z w/o ang wgt')
    axes[2].plot(aligned_vel_angless[:, 2], label='Aligned Estimated Velocity Z w/o ang wgt')
    axes[2].plot(vel_gt[:, 2], label='Ground Truth Velocity Z', linestyle='dashed')
    axes[2].set_ylabel('Velocity Z (m/s)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
