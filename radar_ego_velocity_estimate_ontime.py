import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve, lstsq

# Global variables to store data for plotting
timestamps = []
velocities_x = []
velocities_y = []
velocities_z = []

def callback(msg):
    global timestamps, velocities_x, velocities_y, velocities_z
    # Convert ROS PointCloud2 message to numpy array
    field_names = ["x", "y", "z", "intensity", "velocity"]
    points = pc2.read_points(msg, field_names=field_names, skip_nans=True)
    radar_data = []
    doppler_data = []

    for point in points:
        radar_data.append([point[0], point[1], point[2], point[3]])  # x, y, z, intensity
        doppler_data.append(point[4])  # doppler
    
    radar_data = np.array(radar_data)
    doppler_data = np.array(doppler_data)

    # Compute velocities from radar data
    success, v_r, P_v_r, inlier = radar_ego_velocity_estimate(radar_data, doppler_data, config)
    if success:
        velocities_x.append(v_r[0])
        velocities_y.append(v_r[1])
        velocities_z.append(v_r[2])
        timestamps.append(rospy.get_time())

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
    if len(valid_targets) > 10:
        radar_data = np.array(valid_targets)[:, [5, 6, 7, 8]]
        success, v_r, P_v_r, inlier_idx_best = solve_3d_lsq_irls(radar_data, config)
        if success:
            return True, v_r, P_v_r, [valid_targets[idx] for idx in inlier_idx_best]
    return False, [np.nan, np.nan, np.nan], None, None

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

def animate(i):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    ax1.plot(timestamps, velocities_x, label='X Velocity')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('X Velocity')
    ax1.legend()
    
    ax2.plot(timestamps, velocities_y, label='Y Velocity', color='orange')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Y Velocity')
    ax2.legend()
    
    ax3.plot(timestamps, velocities_z, label='Z Velocity', color='green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Z Velocity')
    ax3.legend()

if __name__ == '__main__':
    rospy.init_node('radar_velocity_estimator', anonymous=True)
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

    radar_topic = '/ti_mmwave/radar_scan_pcl_0'
    rospy.Subscriber(radar_topic, PointCloud2, callback)

    # Setup matplotlib for real-time plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    ani = FuncAnimation(fig, animate, interval=1000)
    plt.tight_layout()
    plt.show()

    rospy.spin()
