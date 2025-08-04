import os, sys
import numpy as np
import torch
import cv2
from PIL import Image
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from collision_detector import ModelFreeCollisionDetector


class GraspDetector:
    def __init__(self, checkpoint_path, num_point=20000, num_view=300):
        self.num_point = num_point
        # 初始化模型
        self.net = GraspNet(
            input_feature_dim=0,
            num_view=num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False
        )

        # 加载模型权重
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.net.to(device)
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        print(f"Model loaded from {checkpoint_path}")

    def process_rgb_only(self, rgb_path):
        """仅处理RGB图像"""
        print("Using RGB only mode")
        # 读取图像
        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        h, w = color.shape[:2]

        # 生成伪点云
        x = np.linspace(-0.5, 0.5, w)
        y = np.linspace(-0.5 * h / w, 0.5 * h / w, h)
        xv, yv = np.meshgrid(x, y)
        points = np.stack([xv, yv, np.zeros_like(xv)], axis=2)

        points = points.reshape(-1, 3)
        colors = color.reshape(-1, 3)

        # 采样点
        if len(points) >= self.num_point:
            idxs = np.random.choice(len(points), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(points))
            idxs2 = np.random.choice(len(points), self.num_point - len(points), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        points_sampled = points[idxs]
        colors_sampled = colors[idxs]

        batch_data = {
            'point_clouds': torch.from_numpy(points_sampled).float().unsqueeze(0).to(self.device),
            'cloud_colors': torch.from_numpy(colors_sampled).float().unsqueeze(0).to(self.device),
        }

        return batch_data, points_sampled

    def process_rgbd(self, rgb_path, depth_path, camera_intrinsic, depth_factor):
        """处理RGB-D图像"""
        print("Using RGB-D mode")
        # 读取图像
        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))

        # 相机参数
        camera = CameraInfo(
            1280.0, 720.0,
            camera_intrinsic[0][0], camera_intrinsic[1][1],
            camera_intrinsic[0][2], camera_intrinsic[1][2],
            depth_factor
        )

        # 生成点云
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # 获取有效点
        depth_mask = (depth > 0)
        cloud_masked = cloud[depth_mask]
        color_masked = color[depth_mask]

        # 采样点
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        batch_data = {
            'point_clouds': torch.from_numpy(cloud_sampled).float().unsqueeze(0).to(self.device),
            'cloud_colors': torch.from_numpy(color_sampled).float().unsqueeze(0).to(self.device),
        }

        return batch_data, cloud_sampled

    def inference(self, batch_data, cloud_sampled=None, collision_thresh=0.01):
        """执行推理"""
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points)

        # 获取抓取结果
        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        # 碰撞检测(仅在有点云数据时进行)
        if cloud_sampled is not None and collision_thresh > 0:
            mfcdetector = ModelFreeCollisionDetector(cloud_sampled, voxel_size=0.01)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]

        return gg

    def visualize_grasps(self, rgb_img, grasps, num_show=10):
        """可视化抓取位姿"""
        img = rgb_img.copy()
        h, w = img.shape[:2]

        # 将归一化坐标转换为图像坐标
        scale_x = w / 2
        scale_y = h / 2
        center_x = w / 2
        center_y = h / 2

        for i in range(min(num_show, len(grasps.translations))):
            # 获取抓取参数
            center = grasps.translations[i][:2]
            rotation = grasps.rotation_matrices[i]
            width = grasps.widths[i] * min(w, h) / 2
            score = grasps.scores[i]

            # 转换为图像坐标
            center_px = np.array([
                center_x + center[0] * scale_x,
                center_y + center[1] * scale_y
            ]).astype(np.int32)

            # 绘制抓取中心点
            cv2.circle(img, tuple(center_px), 5, (0, 255, 0), -1)

            # 绘制抓取方向
            direction = rotation[:2, 0]
            direction = direction / np.linalg.norm(direction)
            end_point = center_px + (direction * 30).astype(np.int32)
            cv2.line(img, tuple(center_px), tuple(end_point), (255, 0, 0), 2)

            # 绘制抓取宽度
            normal = np.array([-direction[1], direction[0]])
            width_points = [
                center_px + (normal * width).astype(np.int32),
                center_px - (normal * width).astype(np.int32)
            ]
            cv2.line(img, tuple(width_points[0]), tuple(width_points[1]), (0, 0, 255), 2)

            # 显示分数
            cv2.putText(img, f'{score:.2f}', tuple(center_px + np.array([10, 10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return img


def main():
    parser = argparse.ArgumentParser(description='GraspNet推理脚本')
    parser.add_argument('--checkpoint', required=True, help='模型检查点路径')
    parser.add_argument('--rgb', required=True, help='RGB图像路径')
    parser.add_argument('--depth', help='深度图路径(可选)')
    parser.add_argument('--camera_matrix', help='相机内参文件路径(.npy)')
    parser.add_argument('--depth_factor', type=float, default=1000.0, help='深度图缩放因子')
    parser.add_argument('--output', default='grasp_result.png', help='输出图像路径')
    parser.add_argument('--num_point', type=int, default=20000, help='采样点数')
    parser.add_argument('--num_view', type=int, default=300, help='视图数')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='碰撞检测阈值')
    args = parser.parse_args()

    # 初始化检测器
    detector = GraspDetector(args.checkpoint, args.num_point, args.num_view)

    # 读取RGB图像
    rgb_img = cv2.imread(args.rgb)
    if rgb_img is None:
        raise ValueError(f"无法读取RGB图像: {args.rgb}")

    # 根据是否有深度图选择处理模式
    if args.depth is not None and args.camera_matrix is not None:
        # RGB-D模式
        camera_matrix = np.load(args.camera_matrix)
        batch_data, cloud_sampled = detector.process_rgbd(
            args.rgb, args.depth, camera_matrix, args.depth_factor
        )
    else:
        # 仅RGB模式
        batch_data, cloud_sampled = detector.process_rgb_only(args.rgb)
        cloud_sampled = None  # 仅RGB模式不进行碰撞检测

    # 执行推理
    grasp_group = detector.inference(batch_data, cloud_sampled, args.collision_thresh)
    print(grasp_group)
    # 可视化结果
    vis_img = detector.visualize_grasps(rgb_img, grasp_group)

    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # # 保存结果
    # cv2.imwrite(args.output, vis_img)
    # print(f"结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
