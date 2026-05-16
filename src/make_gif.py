"""
从 woxmn.py 输出的密度帧 PNG 生成 GIF 动图。

用法：
    python make_gif.py                          # 默认生成 GIF
    python make_gif.py --fps 10                 # 控制帧率
    python make_gif.py --loop 0                 # 无限循环
    python make_gif.py --output animation.gif   # 自定义输出名

依赖：
    pip install imageio imageio-ffmpeg
"""

import argparse
import sys
from pathlib import Path

try:
    import imageio.v3 as iio
except ImportError:
    print("需要安装 imageio: pip install imageio imageio-ffmpeg")
    sys.exit(1)


def create_gif(
    frame_dir: str = "orszagtang_outputs",
    output: str = "orszagtang_evolution.gif",
    fps: int = 4,
    loop: int = 0,
    sort_key: str = "time",
):
    """
    将帧 PNG 合成 GIF 动图。

    参数
    ----------
    frame_dir : str
        存放 frame_t=*.png 的目录
    output : str
        输出的 GIF 文件名
    fps : int
        帧率（帧/秒），默认 4
    loop : int
        循环次数，0=无限循环
    sort_key : str
        排序方式: "time"(按时间) 或 "name"(按文件名)
    """
    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        print(f"错误：目录不存在 {frame_dir}")
        return

    # 收集所有 frame_t=*.png 文件
    frames = sorted(frame_dir.glob("frame_t=*.png"))

    if not frames:
        print(f"错误：在 {frame_dir} 中未找到 frame_t=*.png 文件")
        return

    if sort_key == "time":
        # 按文件名中的时间数值排序
        def _time_key(p):
            try:
                return float(p.stem.replace("frame_t=", ""))
            except ValueError:
                return 0.0

        frames.sort(key=_time_key)

    print(f"发现 {len(frames)} 帧:")
    for f in frames:
        print(f"  {f.name}")

    # 读取全部帧
    images = []
    for f in frames:
        images.append(iio.imread(f))

    # 写入 GIF
    duration = 1000 / fps  # 每帧显示毫秒数
    iio.imwrite(
        output,
        images,
        duration=duration,
        loop=loop,
    )

    print(f"\nGIF 已保存: {output}")
    print(f"  帧数: {len(images)}")
    print(f"  帧率: {fps} FPS")
    print(f"  总时长: {len(images) / fps:.1f} 秒")
    print(f"  循环: {'无限' if loop == 0 else loop} 次")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从帧 PNG 合成 MHD 模拟 GIF")
    parser.add_argument("--frame-dir", default="orszagtang_outputs", help="帧 PNG 所在目录")
    parser.add_argument("--output", "-o", default="orszagtang_evolution.gif", help="输出 GIF 文件名")
    parser.add_argument("--fps", type=int, default=4, help="帧率 (默认 4)")
    parser.add_argument("--loop", type=int, default=0, help="循环次数 (0=无限)")
    parser.add_argument("--sort", choices=["time", "name"], default="time", help="排序方式")
    args = parser.parse_args()

    create_gif(
        frame_dir=args.frame_dir,
        output=args.output,
        fps=args.fps,
        loop=args.loop,
        sort_key=args.sort,
    )
