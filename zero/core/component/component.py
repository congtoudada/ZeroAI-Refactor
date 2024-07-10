import os
from abc import ABC
from loguru import logger
import json
from zero.core.info.base_info import BaseInfo
from zero.core.key.shared_key import SharedKey
from zero.utility.log_kit import LogKit


class Component(ABC):
    """
    所有组件的基类
    """
    def __init__(self, shared_memory: dict):
        self.enable = True  # 是否启用
        self.shared_memory: dict = shared_memory  # 共享内存
        self.config: BaseInfo = None  # 配置文件
        self.pname = f"[ {os.getpid()}:component ]"  # pname
        self.esc_event = None  # 退出事件
        self.is_child = False  # 是否为子组件
        self.children = []  # 子组件列表

    def on_start(self):
        # 绑定退出事件（根组件绑定即可）
        if not self.is_child and self.shared_memory is not None:
            self.esc_event = self.shared_memory[SharedKey.EVENT_ESC]
        # 初始化日志模块，只有root组件才需要配置
        if not self.is_child:
            if not LogKit.load_info(self.config):
                logger.info(f"{self.pname} 日志模块被关闭!")
        # 转换为带缩进的JSON字符串并输出
        json_string = json.dumps(self.config.__dict__, indent=4)
        logger.info(f"{self.pname} 配置文件参数: \n{json_string}")

    def on_update(self) -> bool:
        return True

    def on_destroy(self):
        logger.info(f"{self.pname} destroy!")

    def add_component(self, component):
        if isinstance(component, Component):
            component.is_child = True
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
