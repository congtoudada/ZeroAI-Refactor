import os
import time
import cv2
from UltraDict import UltraDict
from loguru import logger

from simple_http.simple_http_helper_info import SimpleHttpHelperInfo
from simple_http.simple_http_key import SimpleHttpKey
from zero.utility.config_kit import ConfigKit


class SimpleHttpHelper:
    def __init__(self, config):
        if config is None or config == "":
            return
        self.config: SimpleHttpHelperInfo = SimpleHttpHelperInfo(ConfigKit.load(config))
        self.shared_memory = UltraDict(name=self.config.output_port)

    def get(self, uri: str):
        req_package = {
            SimpleHttpKey.HTTP_PACKAGE_URI.name: uri,
            SimpleHttpKey.HTTP_PACKAGE_METHOD.name: 1,
            SimpleHttpKey.HTTP_PACKAGE_JSON.name: None
        }
        if not self.shared_memory.__contains__(self.config.output_port):
            logger.error(f"发送Http请求失败, 没有开启SimpleHttpService! port: {self.config.output_port}")
            return
        self.shared_memory[self.config.output_port].put(req_package)

    def post(self, uri: str, data: dict):
        req_package = {
            SimpleHttpKey.HTTP_PACKAGE_URI.name: uri,
            SimpleHttpKey.HTTP_PACKAGE_METHOD.name: 2,
            SimpleHttpKey.HTTP_PACKAGE_JSON.name: data
        }
        if not self.shared_memory.__contains__(self.config.output_port):
            logger.error("发送Http请求失败, 没有开启SimpleHttpService!")
            return
        self.shared_memory[self.config.output_port].put(req_package)

    def send_warn_result(self, pname, output_dir, camId, warnType, per_id, shot_img, img_enable, web_enable):
        # 导出图
        time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        warn_str = ""
        if warnType == 1:
            warn_str = "Phone"
        elif warnType == 2:
            warn_str = "Helmet"
        elif warnType == 3:
            warn_str = "Card"
        elif warnType == 4:
            warn_str = "Intrude"
        img_path = os.path.join(output_dir, f"{time_str}_{warn_str}_{per_id}.jpg")
        if img_enable and shot_img is not None:
            cv2.imwrite(img_path, shot_img)
            logger.info(f"{pname}存图成功，路径: {img_path}")
        if web_enable:
            # 通知后端
            data = {
                "recordTime": time_str,
                "camId": camId,
                "warnType":  warnType,
                "personId": per_id,
                "shotImg": img_path
            }
            self.post("warn", data)
