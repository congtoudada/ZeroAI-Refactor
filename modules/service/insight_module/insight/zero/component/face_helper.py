import os
from typing import Dict
from loguru import logger

from insight.zero.component.face_process_helper import FaceProcessHelper
from insight.zero.info.face_helper_info import FaceHelperInfo
from zero.utility.config_kit import ConfigKit


class FaceHelper:
    """
    人脸识别帮助类
    """
    def __init__(self, shared_memory, config, cam_id, callback):
        super().__init__(shared_memory)
        if isinstance(config, str):
            self.config: FaceHelperInfo = FaceHelperInfo(ConfigKit.load(config))
        else:
            self.config: FaceHelperInfo = config
        self.pname = f"[ {os.getpid()}:face_helper ]"
        self.handler = FaceProcessHelper(shared_memory, self.config, cam_id, self.face_callback)
        # key: obj_id
        # value: { "per_id": 1, "score": 0 }
        self.face_dict: Dict[int, dict] = {}  # 人脸识别结果集
        self.callback = callback

    def tick(self):
        """
        tick: 用于轮询响应队列
        :return:
        """
        self.handler.tick()

    def can_send(self, obj_id, diff, per_y) -> bool:
        """
        是否可以发送
        :param obj_id:
        :param diff:
        :param per_y:
        :return:
        """
        if self.face_dict.__contains__(obj_id) and self.face_dict[obj_id]['per_id'] != 1:  # 不是陌生人，不发送
            return False
        if not diff > self.config.face_min_send_interval:  # 不满足发送间隔，不发送
            return False
        if not self.config.face_cull_up_y < per_y < 1.0 - self.config.face_cull_down_y:  # 在剔除区域，不发送
            return False
        return True

    def send(self, obj_id, image):
        """
        发送人脸识别请求（内部还有判断一次）
        :param obj_id:
        :param image:
        :return:
        """
        self.handler.send(obj_id, image)

    def destroy_obj(self, obj_id):
        """
        清除对象
        :param obj_id:
        :return:
        """
        if self.face_dict.__contains__(obj_id):
            self.face_dict.pop(obj_id)

    def face_callback(self, obj_id, per_id, score):
        logger.info(f"{self.pname} 收到人脸响应: {obj_id} {per_id} {score}")
        # 添加到结果集缓存
        self.face_dict[obj_id] = {
            "per_id": per_id,
            "score": score
        }
        # 触发外界回调函数
        if self.callback is not None:
            self.callback(obj_id, per_id, score)
