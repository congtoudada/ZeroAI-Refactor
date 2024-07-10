from enum import Enum


class SharedKey(Enum):
    """
       单个摄像头进程内，摄像头与算法共享数据的Key常量
       """
    """
    全局
    """
    EVENT_ESC = 0  # 退出事件
    LAUNCH_COUNTER = 1  # 启动计数器（用于控制初始化各模块的顺序）
