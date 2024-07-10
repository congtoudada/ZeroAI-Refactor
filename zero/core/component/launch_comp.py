import multiprocessing
import os
import pickle
import signal
import sys
import time
from multiprocessing import Process

import cv2
from UltraDict import UltraDict
from loguru import logger
from zero.core.key.shared_key import SharedKey

from zero.core.component.component import Component
from zero.core.info.launch_info import LaunchInfo
from zero.utility.config_kit import ConfigKit


class LaunchComponent(Component):
    """
    LauncherComponent: 算法启动入口组件
    """
    def __init__(self, application_path: str):
        super().__init__(None)  # LaunchComponent作为主进程，是共享内存的持有者，无需存
        self.config: LaunchInfo = LaunchInfo(ConfigKit.load(application_path))
        self.pname = f"[ {os.getpid()}:main ]"  # 重写父类pname
        self.shared_memory = None  # 共享内存
        self.esc_event = None  # 退出事件

    def on_start(self):
        """
        初始化时调用一次
        :return:
        """
        super().on_start()
        # 设置子进程开启方式
        if sys.platform.startswith('linux'):  # linux默认fork，但fork可能不支持cuda
            multiprocessing.set_start_method('spawn')
        # 初始化退出信号事件
        self.esc_event = multiprocessing.Manager().Event()
        # 注册终止信号 Ctrl+C可以触发
        signal.signal(signal.SIGINT, self.handle_termination)
        signal.signal(signal.SIGTERM, self.handle_termination)

        # -------------------------------- 1.初始化共享内存 --------------------------------
        # self.shared_memory: dict = multiprocessing.Manager().dict()
        self.shared_memory = UltraDict(shared_lock=True)  # 加锁版本（稳 > 快）
        self.shared_memory[SharedKey.EVENT_ESC] = self.esc_event
        self.shared_memory[SharedKey.LAUNCH_COUNTER] = 0

        # -------------------------------- 2.初始化全局服务 --------------------------------


    def handle_termination(self, signal_num, frame):
        print(f'接收到信号 {signal_num}, 开始清理并退出...')
        self.esc_event.set()
