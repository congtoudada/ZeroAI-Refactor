import json
import multiprocessing
import os
import time

import requests
from UltraDict import UltraDict
from loguru import logger

from simple_http.simple_http_info import SimpleHttpInfo
from simple_http.simple_http_key import SimpleHttpKey
from zero.core.component.component import Component
from zero.core.key.global_key import GlobalKey
from zero.utility.config_kit import ConfigKit


class SimpleHttpComponent(Component):
    """
    简单的Http服务
    """
    SHARED_MEMORY_NAME = "simple_http"

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: SimpleHttpInfo = SimpleHttpInfo(ConfigKit.load(config_path))  # 配置文件内容
        self.http_shared_memory = UltraDict(name=SimpleHttpComponent.SHARED_MEMORY_NAME)
        self.pname = f"[ {os.getpid()}:simple http ]"
        self.req_queue = None  # 后端请求队列
        self.headers = {
            'content-type': 'application/json;charset=utf-8'
        }

    def on_start(self):
        super().on_start()
        # 初始化请求缓存
        self.req_queue = multiprocessing.Manager().Queue()
        self.http_shared_memory[SimpleHttpKey.HTTP_REQ.name] = self.req_queue

    def on_update(self) -> bool:
        # 处理请求
        while not self.req_queue.empty():
            req_package = self.req_queue.get()
            uri = req_package[SimpleHttpKey.HTTP_PACKAGE_URI.name]
            method = req_package[SimpleHttpKey.HTTP_PACKAGE_METHOD.name]
            content = req_package[SimpleHttpKey.HTTP_PACKAGE_JSON.name]
            full_url = self._get_full_url(uri)
            response = None
            try:
                if method == 1:  # GET
                    logger.info(f"{self.pname} 发送Get请求，路径: {full_url}")
                    response = requests.get(self._get_full_url(uri))
                elif method == 2:
                    logger.info(f"{self.pname} 发送Post请求，路径: {full_url}")
                    response = requests.post(self._get_full_url(uri), headers=self.headers, data=json.dumps(content))
            except Exception as e:
                logger.error(f"{self.pname} {e}")
            if response is not None:
                if response.status_code == 200:
                    logger.info(f"{self.pname} 成功收到后端响应，路径: {full_url}")
                else:
                    logger.error(f"{self.pname} 请求失败[{response.status_code}]，没有收到后端响应，路径: {full_url}")

        return False

    def _get_full_url(self, uri: str) -> str:
        return f"http://{self.config.web_address}/algorithm/{uri}"

    def on_destroy(self):
        self.http_shared_memory.unlink()
        super().on_destroy()

def create_process(shared_memory, config_path: str):
    comp = SimpleHttpComponent(shared_memory, config_path)
    comp.start()
    shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
    comp.update()


if __name__ == '__main__':
    url = "http://localhost:8080/unity/camera_list"
    print("开始Get请求")
    response = requests.get(url)
    print(response)

    url = "http://localhost:8080/unity/record_delete"
    headers = {
        'content-type': 'application/json;charset=utf-8'
    }
    data = {
        'key': 10,
        'pageType': 1
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response)
    # url = "http://210.30.97.233:8080/unity/camera_list"
    # print("开始Post请求")
    # WebKit.post(url, {"ReqPersonWarnADTO": {'camId': 10}})
    # WebKit.post(url, {"DTO": 10})

