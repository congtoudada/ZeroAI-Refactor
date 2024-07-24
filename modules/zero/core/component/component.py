import os
import signal
import time
from abc import ABC
from loguru import logger
import json

from zero.core.info.base_info import BaseInfo
from zero.core.key.global_key import GlobalKey
from zero.utility.log_kit import LogKit
from zero.utility.timer_kit import TimerKit


class Component(ABC):
    """
    所有组件的基类
    """
    def __init__(self, shared_memory: dict):
        self.enable = True  # 是否启用
        self.shared_memory: dict = shared_memory  # 全局共享内存
        self.config: BaseInfo = None  # 配置文件
        self.pname = f"[ {os.getpid()}:component ]"
        self.esc_event = None  # 退出事件
        self.is_child = False  # 是否为子组件
        self.has_child = False  # 是否有孩子
        self.children = []  # 子组件列表
        self._update_delay = 1.0 / 60  # 每帧延迟
        self._default_update_delay = 1.0 / 60  # 默认帧延迟（避免反复计算）
        self.update_timer = TimerKit()

    def on_start(self):
        # 绑定退出事件（根组件绑定即可）
        if not self.is_child and self.shared_memory is not None:
            self.esc_event = self.shared_memory[GlobalKey.EVENT_ESC.name]
        # 初始化日志模块，只有root组件才需要配置
        if not self.is_child:
            if not LogKit.load_info(self.config):
                logger.info(f"{self.pname} 日志模块被关闭!")
        # 转换为带缩进的JSON字符串并输出
        json_string = json.dumps(self.config.__dict__, indent=4)
        logger.info(f"{self.pname} {type(self)} 配置文件参数: \n{json_string}")
        self._default_update_delay = 1.0 / self.config.update_fps

    def on_update(self) -> bool:
        self._update_delay = self._default_update_delay
        return True

    def on_destroy(self):
        logger.info(f"{self.pname} destroy!")

    def add_component(self, component):
        if isinstance(component, Component):
            component.is_child = True
            self.has_child = True
            self.children.append(component)

    def get_component(self, class_type):
        """
        获取子组件
        :param class_type:
        :return:
        """
        for child in self.children:
            if isinstance(child, class_type):
                return child
        logger.error(f"{self.pname} 找不到组件: {class_type}")
        return None

    def get_components(self, class_type):
        """
        获取指定类型的子组件
        :param class_type:
        :return:
        """
        ret = []
        for child in self.children:
            if isinstance(child, class_type):
                ret.append(child)
        if len(ret) == 0:
            logger.error(f"{self.pname} 找不到组件: {class_type}")
        return ret

    def pause(self):
        """
        组件暂停运行（根组件暂停，全部暂停）
        :return:
        """
        self.enable = False

    def resume(self):
        """
        组件继续运行
        :return:
        """
        self.enable = True

    def start(self):
        # 组件初始化
        self.on_start()  # 先初始化父组件（通常在该函数内添加相应子组件）
        for child in self.children:  # 再执行子组件更新
            self.has_child = True
            child.on_start()

    def update(self):
        # 收到退出信号
        if self.esc_event.is_set():
            for child in self.children:  # 先销毁子组件
                child.on_destroy()
            self.on_destroy()  # 再销毁父组件
            return
        # 组件更新
        while True:
            if self.enable:
                self.on_update()  # 先执行父组件的更新
                if self.has_child:  # 多一层判断（更省性能？）
                    for child in self.children:  # 再执行子组件更新
                        if child.enable:
                            child.on_update()
            if self.esc_event.is_set():
                for child in self.children:  # 先销毁子组件
                    child.on_destroy()
                self.on_destroy()  # 再销毁父组件
                return
            if self._update_delay != -1:
                time.sleep(self._update_delay)
