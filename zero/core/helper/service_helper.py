import importlib
import os
import sys
import time
from multiprocessing import Process
from loguru import logger

from zero.core.info.launch_info import LaunchInfo
from zero.core.key.shared_key import SharedKey


class ServiceHelper:
    """
    服务帮助类
    """
    def __init__(self, shared_memory, launch_config: LaunchInfo):
        self.shared_memory = shared_memory
        self.config: LaunchInfo = launch_config
        self.pname = f"[ {os.getpid()}:service helper ]"

    def launch_service(self):
        lock = -1
        for comp in self.config.service:
            # 存在依赖关系，必须顺序初始化
            while lock == self.shared_memory[SharedKey.LAUNCH_COUNTER]:
                time.sleep(1)
            lock = self.shared_memory[SharedKey.LAUNCH_COUNTER]
            logger.info(f"{self.pname} 启动服务: {os.path.basename(comp['path']).split('.')[0]}")
            module_file = comp['path']
            if os.path.exists(module_file):
                sys.path.append(os.path.dirname(module_file))
                module = importlib.import_module(os.path.basename(module_file).split(".")[0])
                if module.__dict__.__contains__("create_process"):
                    logger.info(f"{self.pname} 服务配置文件路径: {comp['conf']}")
                    Process(target=module.create_process,
                            args=(self.shared_memory, comp['conf']),
                            daemon=False).start()
                else:
                    logger.error(f"{self.pname} 服务启动失败！没有实现create_process: {os.path.basename(module_file)}")
            else:
                logger.error(f"{self.pname} 服务启动失败！找不到py脚本: {module_file}")
