import os
import rosbag
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import rospy
from scipy.linalg import solve, lstsq
from sklearn.cluster import DBSCAN
import open3d as o3d

save_path = "/home/wbl/code/visual_radar_slam/project_radar2img/result/pc_img/"

def cluster_dynamic_points(dynamic_points, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dynamic_points)
    labels = clustering.labels_
    
    clustered_points = []
    for label in np.unique(labels):
        if label == -1:  # 噪声点
            continue
        cluster = dynamic_points[labels == label]
        centroid = np.mean(cluster, axis=0)
        clustered_points.append(centroid)
    
    return clustered_points

def read_and_process_bag(bag_name, image_topic, pc_topic, camera_intrinsics, camera_extrinsics, config, threshold=10, lidar_topic=None, lidar_extrinsics=None, lidar2radar=None):
    bag = rosbag.Bag(bag_name)
    bridge = CvBridge()

    bag_start_time = bag.get_start_time()
    read_start_time = bag_start_time + 25

    for topic, msg, t in bag.read_messages(topics=[image_topic], start_time=rospy.Time(read_start_time)):
        if topic == image_topic:
            dtype = np.uint8
            if msg.encoding == "rgb8":
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, 3)
            elif msg.encoding == "bgr8":
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, 3)
            elif msg.encoding == "mono8":
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            elif msg.encoding == "bgra8":
                cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, 4)
                cv_image = cv_image[:, :, :3]  # Convert BGRA to BGR
            else:
                raise ValueError(f"Unsupported encoding: {msg.encoding}")
            
            cv_image = np.ascontiguousarray(cv_image)  # Ensure the array is contiguous
            image_timestamp = t.to_sec() + 0.08

            # Find the nearest point cloud message
            min_time_diff = float('inf')
            min_time_diff_lidar = float('inf')
            nearest_pc = None
            pc_topic_list = [pc_topic]
            if lidar_topic is not None:
                pc_topic_list.append(lidar_topic)
            nearest_lidar = None
            time_diff = 0
            time_diff_lidar = 0
            for topic0, pc_msg, pc_time in bag.read_messages(topics=pc_topic_list, start_time=rospy.Time(image_timestamp - 0.2)):
                if topic0 == pc_topic:
                    time_diff = abs(pc_time.to_sec() - image_timestamp)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        pc = pc2.read_points(pc_msg, field_names=("x", "y", "z", "doppler", "intensity"), skip_nans=True)
                        nearest_pc = np.array(list(pc))
                elif topic0 == lidar_topic:
                    time_diff_lidar = abs(pc_time.to_sec() - image_timestamp)
                    if time_diff_lidar < min_time_diff_lidar:
                        min_time_diff_lidar = time_diff_lidar
                        pc = pc2.read_points(pc_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
                        nearest_lidar = np.array(list(pc))

                # Break if the time difference is too large
                if (time_diff > 0.1 and time_diff_lidar > 0.1) or (time_diff > min_time_diff and time_diff_lidar > min_time_diff_lidar):
                    break

            if nearest_lidar is not None:
                project_lidar, project_lidar_image = project_lidar_to_image(nearest_lidar, cv_image.copy(), camera_intrinsics, lidar_extrinsics)
                cv2.imshow("Projected Lidar", project_lidar_image)

            if nearest_pc is not None:
                dynamic_points = find_dynamic_points(nearest_pc, config)
                if len(dynamic_points) > 0:
                    dynamic_points_np = np.array(dynamic_points)
                    # cluster dynamic points
                    clustered_points = cluster_dynamic_points(dynamic_points_np, eps=0.8, min_samples=2)
                    dynamic_points_np = np.array(clustered_points)
                    if len(dynamic_points_np) == 0:
                        cv2.imshow("Segmented Image", cv_image)
                        cv2.imshow("Projected Dynamic Points", cv_image)
                        cv2.waitKey(1)
                        continue

                    # projected_points, pointpro_image = project_point_cloud_to_image(dynamic_points_np, project_lidar_image.copy(), camera_intrinsics, camera_extrinsics)
                    projected_points, pointpro_image = project_point_cloud_to_image(dynamic_points_np, project_lidar_image.copy(), camera_intrinsics, camera_extrinsics)
                    if len(projected_points) > 0:
                        # Add a label to each seed point
                        seed_points = [(int(pt[0]), int(pt[1]), i + 1) for i, pt in enumerate(projected_points)]
                        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                        segmented_image = region_growing(gray_image, seed_points, threshold)
                        if np.max(segmented_image) == 0:
                            cv2.imshow("Segmented Image", cv_image)
                            cv2.imshow("Projected Dynamic Points", cv_image)
                            cv2.waitKey(1)
                            continue

                        segmented_image_display = (segmented_image * (255 // segmented_image.max())).astype(np.uint8)
                        color_segmented_image = cv2.applyColorMap(segmented_image_display, cv2.COLORMAP_JET)
                        overlay_image = cv2.addWeighted(pointpro_image, 0.7, color_segmented_image, 0.3, 0)
                        overlay_image = cv2.addWeighted(project_lidar_image, 0.8, pointpro_image, 0.3, 0)

                        # Overlay segmented regions on the original image
                        color_segmented_image = overlay_segmentation(cv_image, segmented_image)

                        cv2.imshow("Segmented Image", color_segmented_image)
                        cv2.imshow("Projected Dynamic Points", pointpro_image)
                        cv2.waitKey(2)

                        # Save the images and point clouds
                        img_save_path = os.path.join(save_path, f"{image_timestamp}_img.png")
                        radar_pc_save_path = os.path.join(save_path, f"{image_timestamp}_radar.pcd")
                        lidar_pc_save_path = os.path.join(save_path, f"{image_timestamp}_lidar.pcd")
                        
                        cv2.imwrite(img_save_path, color_segmented_image)
                        save_point_cloud(radar_pc_save_path, nearest_pc[:, :3])
                        save_point_cloud(lidar_pc_save_path, nearest_lidar[:, :3])
                    else:
                        cv2.imshow("Segmented Image", cv_image)
                        cv2.imshow("Projected Dynamic Points", cv_image)
                        cv2.waitKey(1)

                    # delete lidar dynamic points by radar dynamic points, radius 0.5m
                    for radar_point in dynamic_points_np:
                        radar_point_inlidar = (lidar2radar @ np.hstack((radar_point, 1)))[:3]
                        # Calculate distances from this radar point to all lidar points
                        distances = np.linalg.norm(nearest_lidar[:, :3] - radar_point_inlidar, axis=1)
                        # Find lidar points within 0.5m radius
                        mask = distances > 1
                        # Keep only points outside the radius
                        nearest_lidar = nearest_lidar[mask]
                    
                    # Save filtered lidar points
                    filtered_lidar_save_path = os.path.join(save_path, f"{image_timestamp}_filtered_lidar.pcd")
                    save_point_cloud(filtered_lidar_save_path, nearest_lidar[:, :3])

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                cv2.imshow("Segmented Image", cv_image)
                cv2.imshow("Projected Dynamic Points", cv_image)
                cv2.waitKey(1)

    bag.close()
    cv2.destroyAllWindows()

def find_dynamic_points(point_cloud, config):
    v_dopplers = point_cloud[:, 3]
    success, v_r, P_v_r, inliers, outliers = radar_ego_velocity_estimate(point_cloud, v_dopplers, config)
    if success:
        return [[pt[0], pt[1], pt[2]] for pt in outliers]
    return []

def project_point_cloud_to_image(point_cloud, image, camera_intrinsics, camera_extrinsics):
    R = camera_extrinsics[:3, :3]
    t = camera_extrinsics[:3, 3]

    projected_points = []
    for point in point_cloud:
        point_camera = R @ point[:3] + t
        point_image = camera_intrinsics @ point_camera
        u = point_image[0] / point_image[2]
        v = point_image[1] / point_image[2]
        projected_points.append((int(u), int(v)))

    depth = np.linalg.norm(point_cloud[:, :3], axis=1)
    projected_points2 = []
    for i, point in enumerate(projected_points):
        if 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0] and 1 < depth[i] < 30:
            projected_points2.append(projected_points[i])
            # color = intensity_to_color(depth[i], 1, 30)
            color = (255, 255, 255)
            cv2.circle(image, point, 5, color, -1)

    return projected_points2, image

def project_lidar_to_image(point_cloud, image, camera_intrinsics, camera_extrinsics):
    R = camera_extrinsics[:3, :3]
    t = camera_extrinsics[:3, 3]

    projected_points = []
    for point in point_cloud:
        point_camera = R @ point[:3] + t
        point_image = camera_intrinsics @ point_camera
        u = point_image[0] / point_image[2]
        v = point_image[1] / point_image[2]
        projected_points.append((int(u), int(v)))

    depth = np.linalg.norm(point_cloud[:, :3], axis=1)
    projected_points2 = []
    for i, point in enumerate(projected_points):
        if 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0] and 1 < depth[i] < 30:
            projected_points2.append(projected_points[i])
            color = intensity_to_color(depth[i], 0, 30)
            cv2.circle(image, point, 2, color, -1)

    return projected_points2, image

def intensity_to_color(intensity, min_intensity=0.0, max_intensity=255.0):
    if intensity <= 0:
        intensity = min_intensity
    log_intensity = np.log1p(intensity - min_intensity)
    log_max_intensity = np.log1p(max_intensity - min_intensity)
    
    normalized_intensity = log_intensity / log_max_intensity
    normalized_intensity = np.clip(normalized_intensity, 0, 1)
    
    r = int(255 * normalized_intensity)
    g = 0
    b = int(255 * (1 - normalized_intensity))
    
    return (b, g, r)

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
    if len(valid_targets) > 2:
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
        return False, None, None, None

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

def region_growing(image, seed_points, threshold=10):
    segmented_image = np.zeros(image.shape, dtype=np.int32)
    
    for seed in seed_points:
        x, y, lbl = seed
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue
        if segmented_image[y, x] == 0:  # 避免重复标记
            segmented_image[y, x] = lbl
            queue = [(x, y)]
            region_sum = int(image[y, x])
            region_size = 1

            while queue:
                cx, cy = queue.pop(0)
                region_mean = region_sum / region_size

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy

                        if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and segmented_image[ny, nx] == 0:
                            if abs(int(image[ny, nx]) - region_mean) < threshold:
                                segmented_image[ny, nx] = lbl
                                queue.append((nx, ny))
                                region_sum += int(image[ny, nx])
                                region_size += 1

    return segmented_image

def overlay_segmentation(image, segmented_image):
    color_segmented_image = image.copy()
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Yellow
    ]
    
    for lbl in np.unique(segmented_image):
        if lbl == 0:
            continue
        mask = (segmented_image == lbl)
        color = colors[lbl % len(colors)]
        color_segmented_image[mask] = color

    return color_segmented_image

def save_point_cloud(file_path, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pc)

def main(bag_name, image_topic, pc_topic, camera_intrinsics, camera_extrinsics, config, threshold=10, lidar_topic=None, lidar_extrinsics=None, lidar2radar=None):
    read_and_process_bag(bag_name, image_topic, pc_topic, camera_intrinsics, camera_extrinsics, config, threshold, lidar_topic, lidar_extrinsics, lidar2radar)

if __name__ == "__main__":
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

    bag_name = '/home/wbl/code/visual_radar_slam/project_radar2img/data1_aligned.bag'
    image_topic = '/zed2i/zed_node/left/image_rect_color'
    pc_topic = '/ars548'
    lidar_topic = '/hesai/pandar'

    camera_intrinsics = np.array([
        [266.519989, 0.0, 329.510010],
        [0.0, 266.802490, 182.491257],
        [0.0, 0.0, 1.0]
    ])

    imu2cam = np.array([
        [0.0178199, 0.00393081, 0.999833, -0.06227199],
        [-0.999839, -0.00199067, 0.0178278, 0.0173767],
        [0.00206042, -0.99999, 0.0038947, -0.06345778],
        [0, 0, 0, 1]
    ])

    imu2lidar = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.14],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # imu2lidar = np.array([
    #     [-0.019438, -0.999, 0.0, 0.0],
    #     [0.999106, -0.011923, 0.0, 0.0],
    #     [0.0, 0.0, 1.0, 0.14],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])

    # imu2lidar = np.array([
    #     [-0.099438, -0.992995, -0.063813, -0.012713],
    #     [0.994106, -0.101923,  0.036942, -0.00129],
    #     [-0.043187, -0.059763,  0.997278, 0.184497],
    #     [0, 0, 0, 1]
    # ])

    imu2radar = np.array([
        [0.9924039, -0.0868241, -0.0871557, 0.0],
        [0.0871557,  0.9961947,  0.0000000, 0.0],
        [0.0868241, -0.0075961,  0.9961947, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    imu2radar = np.array([
        [ 0.99314173, -0.06677885, -0.09596921, 0],
        [ 0.06497833,  0.9976492,  -0.02176915, 0],
        [ 0.09719732,  0.01538394,  0.99514623, 0.22],
        [ 0.,          0.,          0.,          1.]
    ])

    lidar2radar = np.array([
        [-0.046259332448, 0.998929440975, 0.000000000000, 0.099917128682],
        [-0.998929440975, -0.046259332448, 0.000000000000, 0.039892427623],
        [0.000000000000, 0.000000000000, 1.000000000000, 0.547123491764],
        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
    ])

    cam2lidar = np.linalg.inv(imu2cam) @ imu2lidar
    cam2radar = np.linalg.inv(imu2cam) @ imu2radar
    cam2radar = cam2lidar @ lidar2radar

    camera_extrinsics = cam2radar

    main(bag_name, image_topic, pc_topic, camera_intrinsics, camera_extrinsics, config, threshold=10, lidar_topic=lidar_topic, lidar_extrinsics=cam2lidar, lidar2radar=lidar2radar)
