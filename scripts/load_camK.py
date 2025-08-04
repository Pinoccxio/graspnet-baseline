import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='保存相机内参矩阵')
    parser.add_argument('--output', type=str, default='./camera_intrinsic.npy', help='输出文件路径')
    parser.add_argument('--matrix', type=str, default='1200,900,640,360', help='相机内参矩阵(可选)')
    parser.add_argument('--fx', type=float, default=1200.0, help='焦距 fx')
    parser.add_argument('--fy', type=float, default=1200.0, help='焦距 fy')
    parser.add_argument('--cx', type=float, default=640.0, help='主点 cx')
    parser.add_argument('--cy', type=float, default=360.0, help='主点 cy')
    args = parser.parse_args()

    if args.matrix:
        try:
            # 分割参数并转换为浮点数
            params = [float(x.strip()) for x in args.params.split(',')]
            if len(params) != 4:
                raise ValueError("需要恰好4个参数值")
            fx, fy, cx, cy = params
        except Exception as e:
            print(f"参数解析错误: {str(e)}")
            print("请使用格式: --matrix fx,fy,cx,cy")
            exit(1)
    elif all([args.fx, args.fy, args.cx, args.cy]):
        # 使用单独参数
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    else:
        print("错误: 必须提供完整的相机内参")
        print("请选择以下一种方式:")
        print("  1) 使用 --params 指定所有参数 (格式: fx,fy,cx,cy)")
        print("  2) 使用 --fx, --fy, --cx, --cy 分别指定参数")
        exit(1)

    print(f'ready to save K')
    # 定义相机内参矩阵 (示例数值)
    intrinsic_matrix = np.array([
        [fx,  0.0,  cx],
        [0.0,  fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # 保存到.npy文件
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, intrinsic_matrix)
    print(f'Successfully save K into {args.output}')


if __name__ == "__main__":
    main()
