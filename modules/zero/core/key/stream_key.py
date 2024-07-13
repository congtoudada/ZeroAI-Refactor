from enum import Enum


class StreamKey(Enum):
    """
    视频流共享内存Key
    使用举例: StreamKey.STREAM_PORT.name（非端口相关不用）
    """
    # STREAM_PORT = 0  # Stream端口（由用户指定）
    STREAM_PACKAGE_FRAME_ID = 1  # 帧ID
    STREAM_PACKAGE_FRAME = 2  # 帧图像
    STREAM_CAM_ID = 3  # 摄像头id
    STREAM_WIDTH = 4  # 摄像头宽
    STREAM_HEIGHT = 5  # 摄像头高
    STREAM_FPS = 6  # 摄像头帧率

