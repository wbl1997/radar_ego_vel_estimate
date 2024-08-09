import os
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.linalg import lstsq
from scipy.spatial.transform import Rotation as R
# from scipy.linalg import cho_factor, cho_solve

def read_tum_poses(file_path):
    """
    Read poses from TUM format pose file.
    Each line in the file should contain: timestamp tx ty tz qx qy qz qw
    """
    data = np.loadtxt(file_path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    orientations = data[:, 4:8]
    return timestamps, positions, orientations

def read_tum_vel(file_path):
    """
    Read poses from TUM format pose file.
    Each line in the file should contain: timestamp tx ty tz qx qy qz qw vx vy vz
    """
    data = np.loadtxt(file_path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    orientations = data[:, 4:8]
    velocities = data[:, 8:11]
    return timestamps, velocities

def compute_velocities(timestamps, positions):
    """
    Compute velocities by differentiating positions.
    """
    dt = np.diff(timestamps)
    velocities = np.diff(positions, axis=0) / dt[:, None]
    return velocities, timestamps[1:]

def compute_and_transform_velocities(timestamps, positions, orientations=None):
    """
    Compute velocities by differentiating positions.
    Optionally transform the velocities to the body frame using given orientations.
    """
    dt = np.diff(timestamps)
    velocities = np.diff(positions, axis=0) / dt[:, None]

    if orientations is not None:
        body_velocities = []
        for v, q in zip(velocities, orientations[:-1]):  # Use orientations corresponding to velocity timestamps
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
                radar_scan.append([point[0], point[1], point[2], point[4]])  # x, y, z, intensity
                doppler_scan.append(point[3])  # doppler
            elif field_names[4] == 'doppler' or field_names[4] == 'velocity':
                radar_scan.append([point[0], point[1], point[2], point[3]])
                doppler_scan.append(point[4])
        radar_data.append(np.array(radar_scan))
        doppler_data.append(np.array(doppler_scan))
        timestamps.append(t.to_sec())
    bag.close()
    return radar_data, doppler_data, timestamps

def read_ground_truth(folder):
    ground_truth = []
    timestamps = []
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            data = np.loadtxt(os.path.join(folder, file))
            timestamps.append(data[:, 0])
            ground_truth.append(data[:, 1:])
    return np.concatenate(ground_truth), np.concatenate(timestamps)

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
            # v_r = cho_solve(cho_factor(HTH), HTy)
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
    # print('Number of valid targets: ', len(valid_targets), 'mean doppler: ', np.mean(np.abs(v_dopplers)))
    if len(valid_targets) > 2:
        v_dopplers_abs = [np.abs(v[8]) for v in valid_targets]
        n = int(len(v_dopplers_abs) * (1.0 - config['allowed_outlier_percentage']))
        median = np.partition(v_dopplers_abs, n)[n]

        if median < config['thresh_zero_velocity']:
            v_r = np.zeros(3)
            P_v_r = np.eye(3) * np.array([config['sigma_zero_velocity_x'], config['sigma_zero_velocity_y'], config['sigma_zero_velocity_z']])**2
            radar_scan_inlier = [target for target in valid_targets if np.abs(target[8]) < 0.5 * config['thresh_zero_velocity']]
            return True, v_r, P_v_r, radar_scan_inlier
        else:
            radar_data = np.array(valid_targets)[:, [5, 6, 7, 8]]
            if config['angle_wgt']:
                weight_x_vec = np.exp(-1*(np.abs(np.array(valid_targets))[:, 0]/(np.pi/2)))
                weight_y_vec = np.exp(1*(np.abs(np.array(valid_targets))[:, 0]/(np.pi/2)))
                weight_z_vec = np.exp(5*(np.abs(np.array(valid_targets))[:, 1]/(np.pi/2)))
                weight_x_mat = np.diag(weight_x_vec)
                weight_y_mat = np.diag(weight_y_vec)
                weight_z_mat = np.diag(weight_z_vec)
                # solve vel_x
                radar_data_x = weight_x_mat @ np.array(valid_targets)[:, [5, 6, 7, 8]]
                success_x, v_r_x, P_v_r_x, inlier_idx_best_x = solve_3d_lsq_irls(radar_data_x, config)
                # solve vel_y
                radar_data_y = weight_y_mat @ np.array(valid_targets)[:, [5, 6, 7, 8]]
                success_y, v_r_y, P_v_r_y, inlier_idx_best_y = solve_3d_lsq_irls(radar_data_y, config)
                # solve vel_z
                radar_data_z = weight_z_mat @ np.array(valid_targets)[:, [5, 6, 7, 8]]
                success_z, v_r_z, P_v_r_z, inlier_idx_best_z = solve_3d_lsq_irls(radar_data_z, config)

                # success, v_r, P_v_r, inlier_idx_best = solve_3d_lsq_irls(radar_data, config)
                # radar_scan_inlier = [valid_targets[idx] for idx in inlier_idx_best]
                success = success_x and success_y and success_z
                if success:
                    v_r = np.array([v_r_x[0], v_r_y[1], v_r_z[2]])
                    P_v_r = np.diag([P_v_r_x[0, 0], P_v_r_y[1, 1], P_v_r_z[2, 2]])
                    inlier_idx_best = inlier_idx_best_x
                    # inlier_idx_best = list(set(inlier_idx_best_x) & set(inlier_idx_best_y) & set(inlier_idx_best_z))
                    radar_scan_inlier = [valid_targets[idx] for idx in inlier_idx_best]
                else:
                    v_r = None
                    P_v_r = None
                    radar_scan_inlier = None
            else:
                success, v_r, P_v_r, inlier_idx_best = solve_3d_lsq_irls(radar_data, config)
                if success:
                    radar_scan_inlier = [valid_targets[idx] for idx in inlier_idx_best]
                else:
                    radar_scan_inlier = None

            return success, v_r, P_v_r, radar_scan_inlier
    else:
        return False, None, None, None

def test_radar_velocity_estimation(bag_name, topic, gt_folder, config, field_names=("x", "y", "z", "doppler", "intensity")):
    radar_data, doppler_data, timestamps = read_bag_data(bag_name, topic, field_names)
    # radar_data = radar_data[300:400]
    # doppler_data = doppler_data[300:400]
    # timestamps = timestamps[300:400]

    print(doppler_data)

    ts_gt, pos_gt, ori_gt = read_tum_poses(gt_folder)
    gt_vel1, gt_ts1 = compute_and_transform_velocities(ts_gt, pos_gt)

    # gt_ts, gt_vel = read_tum_vel(gt_folder)
    
    estimated_velocities = []
    config['angle_wgt'] = False
    for scan, dopplers in zip(radar_data, doppler_data):
        success, v_r, P_v_r, inlier = radar_ego_velocity_estimate(scan, dopplers, config)
        if success:
            estimated_velocities.append(v_r)
            print('Estimated velocity: ', v_r)
        else:
            estimated_velocities.append([np.nan, np.nan, np.nan])
    estimated_velocities_angless = np.array(estimated_velocities)

    estimated_velocities = []
    config['angle_wgt'] = True
    for scan, dopplers in zip(radar_data, doppler_data):
        success, v_r, P_v_r, inlier = radar_ego_velocity_estimate(scan, dopplers, config)
        if success:
            estimated_velocities.append(v_r)
            print('Estimated velocity: ', v_r)
        else:
            estimated_velocities.append([np.nan, np.nan, np.nan])
    estimated_velocities_ang = np.array(estimated_velocities)

    plt.figure()
    # plt.plot(timestamps, estimated_velocities[:, 0], label='Estimated X Velocity')
    # plt.plot(timestamps, estimated_velocities[:, 1], label='Estimated Y Velocity')
    # plt.plot(timestamps, estimated_velocities[:, 2], label='Estimated Z Velocity')

    # plt.plot(timestamps, estimated_velocities_angless[:,0], label='Estimated Norm Velocity w/o ang wgt x')
    # plt.plot(timestamps, estimated_velocities_ang[:,0], label='Estimated Norm Velocity w/ ang wgt x')

    # plt.plot(timestamps, estimated_velocities_angless[:,1], label='Estimated Norm Velocity w/o ang wgt y')
    # plt.plot(timestamps, estimated_velocities_ang[:,1], label='Estimated Norm Velocity w/ ang wgt y')

    # plt.plot(timestamps, estimated_velocities_angless[:,2], label='Estimated Norm Velocity w/o ang wgt z')
    # plt.plot(timestamps, estimated_velocities_ang[:,2], label='Estimated Norm Velocity w/ ang wgt z')

    plt.plot(timestamps, np.linalg.norm(estimated_velocities_angless, axis=1), label='Estimated Norm Velocity w/o ang wgt')
    # plt.plot(timestamps, np.linalg.norm(estimated_velocities_ang, axis=1), label='Estimated Norm Velocity w/ ang wgt')

    # plt.plot(gt_timestamps, ground_truth[:, 0], label='Ground Truth X Velocity', linestyle='dashed')
    # plt.plot(gt_timestamps, ground_truth[:, 1], label='Ground Truth Y Velocity', linestyle='dashed')
    # plt.plot(gt_timestamps, ground_truth[:, 2], label='Ground Truth Z Velocity', linestyle='dashed')
    # plt.plot(gt_ts, np.linalg.norm(gt_vel, axis=1), label='Ground Truth Norm Velocity', linestyle='dashed')
    plt.plot(gt_ts1, np.linalg.norm(gt_vel1, axis=1), label='Ground Truth Norm Velocity', linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.show()

# Example configuration dictionary
config = {
    'irls_iter': 100,
    'use_cholesky_instead_of_bdcsvd': True,
    'inlier_thresh': 0.15,
    'min_dist': 1.0,
    'max_dist': 500.0,
    'azimuth_thresh_deg': 60,
    'elevation_thresh_deg': 40,
    'allowed_outlier_percentage': 0.25,
    'thresh_zero_velocity': 0.05,
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
    # import sys
    # if len(sys.argv) < 2:
    #     print('Usage: python radar_ego_velocity_estimate.py path_to_bag.bag')
    #     sys.exit(1)
    # bag_path = sys.argv[1]
    # if not os.path.exists(bag_path):
    #     print('Bag file does not exist')
    #     sys.exit(1)
    # # radar_topic = '/ars548'
    # # gt_vel_path = '/media/wbl/Elements/data/20240113/data1_aligned/makeGT/scan_states_all.txt'
    # # test_radar_velocity_estimation(bag_path, radar_topic, gt_vel_path, config)

    radar_topic = '/ti_mmwave/radar_scan_pcl_0'
    bag_path = '/media/wbl/Elements/msc/data21_2024-06-18-15-17-51.bag'
    gt_vel_path = '/media/wbl/Elements/msc/data21_gt.txt'
    test_radar_velocity_estimation(bag_path, radar_topic, gt_vel_path, config, field_names=("x", "y", "z", "intensity", "velocity"))