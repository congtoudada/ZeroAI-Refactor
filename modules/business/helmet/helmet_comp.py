import os
import time
from typing import Dict, List
import cv2
import numpy as np
from loguru import logger

from business.common.detection_record import DetectionRecord
from business.common.match_record_helper import MatchRecordHelper
from business.helmet.helmet_info import HelmetInfo
from business.helmet.helmet_item import HelmetItem
from bytetrack.zero.component.bytetrack_helper import BytetrackHelper
from zero.core.component.based_stream_comp import BasedStreamComponent
from zero.core.helper.warn_helper import WarnHelper
from zero.core.key.detection_key import DetectionKey
from zero.core.key.global_key import GlobalKey
from zero.core.key.stream_key import StreamKey
from zero.utility.config_kit import ConfigKit
from zero.utility.img_kit import ImgKit_img_box
from zero.utility.object_pool import ObjectPool


class HelmetComponent(BasedStreamComponent):
    """
    规范佩戴安全帽检测组件
    """
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: HelmetInfo = HelmetInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:helmet for {self.config.input_ports[0]}&{self.config.input_ports[1]} ]"
        self.cam_id = 0
        # key: obj_id value: cls
        self.pool: ObjectPool = ObjectPool(20, HelmetItem)
        self.record_pool: ObjectPool = ObjectPool(20, DetectionRecord)
        self.data_dict: Dict[int, HelmetItem] = {}
        self.border = 10
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 追踪器
        self.helmet_records: List[DetectionRecord] = []  # 安全帽目标检测结果（每帧更新）

    def on_start(self):
        super().on_start()
        self.cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]

    def on_update(self) -> bool:
        self.release_unused()  # 清理无用资源（一定要在最前面调用）
        for i in range(len(self.helmet_records)):
            self.record_pool.push(self.helmet_records[i])
        self.helmet_records.clear()  # 清空安全帽检测记录
        super().on_update()
        return True

    def on_resolve_per_stream(self, read_idx):
        frame, _ = super().on_resolve_per_stream(read_idx)  # 解析视频帧id+视频帧
        if frame is None:  # 没有有效帧
            return frame, None
        # 解析额外数据
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]
        input_det = stream_package[DetectionKey.DET_PACKAGE_RESULT.name]  # 目标检测结果
        return frame, input_det

    def on_process_per_stream(self, idx, frame, input_det):
        """
        处理视频流
        :param idx: 从input_ports[idx]取package
        :param frame: 帧
        :param input_det: 目标检测结果
        :return:
        """
        if input_det is None:
            return None

        if idx == 0:  # 安全帽
            for i, item in enumerate(input_det):
                ltrb = (item[0], item[1], item[2], item[3])
                score = item[4]
                cls = item[5]
                record = self.record_pool.pop()
                record.init(ltrb, score, cls)
                self.helmet_records.append(record)
            return None
        else:  # 人（此时安全帽记录已经填充）
            mot_result = self.tracker.inference(input_det)  # 返回对齐输出后的mot结果
            # 根据mot结果进行安全帽检测
            self._helmet_core(frame, mot_result, self.frame_id_cache[0], frame.shape[1], frame.shape[0])
            return mot_result

    def _helmet_core(self, frame, input_mot, current_frame_id, width, height) -> bool:
        if input_mot is None:
            return
        # 遍历每个人
        for i, obj in enumerate(input_mot):
            ltrb = obj[:4]
            match_idx = MatchRecordHelper.match_bbox(ltrb, [item.ltrb for item in self.helmet_records])
            if match_idx != -1:  # 存在匹配项
                obj_id = int(obj[6])
                helmet = self.helmet_records[match_idx]
                # 安全帽在人里面，更新人的状态
                if not self.data_dict.__contains__(obj_id):  # 没有被记录过
                    item = self.pool.pop()
                    item.init(obj_id, helmet.cls, current_frame_id)
                    self.data_dict[obj_id] = item
                else:  # 已经记录过
                    self.data_dict[obj_id].update(current_frame_id, helmet.cls)
                # 计算结果
                self.process_result(frame, self.data_dict[obj_id], ltrb)  # 满足异常条件就记录

    def process_result(self, frame, helmet_item: HelmetItem, ltrb):
        # 没有报过警且异常状态保持一段时间
        if not helmet_item.has_warn and helmet_item.get_valid_count() > self.config.helmet_valid_count:
            if helmet_item.cls == 0 or helmet_item.cls == 2:
                logger.info(f"安全帽佩戴异常: obj_id{helmet_item.obj_id} cls:{helmet_item.cls}")
                helmet_item.has_warn = True  # 一旦视为异常，则一直为异常，避免一个人重复报警
                shot_img = ImgKit_img_box.draw_img_box(frame, ltrb)
                WarnHelper.send_warn_result(self.pname, self.output_dir, self.cam_id, 2, 1,
                                            shot_img, self.config.stream_export_img_enable,
                                            self.config.stream_web_enable)

    def release_unused(self):
        # 清空长期未更新点
        clear_keys = []
        for key, item in self.data_dict.items():
            if self.frame_id_cache[0] - item.last_update_id > self.config.helmet_lost_frame:
                clear_keys.append(key)
        for key in clear_keys:
            self.pool.push(self.data_dict[key])
            self.data_dict.pop(key)  # 从字典中移除item

    def on_draw_vis(self, idx, frame, input_mot):
        if input_mot is None:  # 检测安全帽的端口，不显示任何内容
            return None
        text_scale = 1
        text_thickness = 1
        line_thickness = 2
        # 标题线
        num = 0 if input_mot is None else input_mot.shape[0]
        cv2.putText(frame, 'inference_fps:%.2f num:%d' %
                    (1. / max(1e-5, self.update_timer.average_time),
                     num), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 对象基准点、包围盒
        if len(self.config.detection_labels) == 0:
            logger.warning(f"{self.pname} detection_labels的长度为0，请在配置文件中配置detection_labels!")
            return frame
        if input_mot is not None:
            for obj in input_mot:
                ltrb = obj[:4]
                obj_id = int(obj[6])
                cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                              color=self._get_color(obj_id), thickness=line_thickness)
                if self.data_dict.__contains__(obj_id):
                    cls = int(self.data_dict[obj_id].cls)
                    cv2.putText(frame, f"{obj_id}:{self.config.detection_labels[cls]}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, self._get_color(obj_id), thickness=text_thickness)
                else:
                    cv2.putText(frame, f"{obj_id}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, self._get_color(obj_id), thickness=text_thickness)
        # 安全帽
        for i, item in enumerate(self.helmet_records):
            ltrb = item.ltrb
            cls = int(item.cls)
            score = item.score
            cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                          color=(0, 0, 255), thickness=line_thickness)
            id_text = f"{self.config.detection_labels[cls]}({score:.2f})"
            cv2.putText(frame, id_text, (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, (0, 0, 255), thickness=text_thickness)
        # 可视化并返回
        return frame

    def _get_color(self, idx):
        idx = (1 + idx) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

def create_process(shared_memory, config_path: str):
    comp: HelmetComponent = HelmetComponent(shared_memory, config_path)  # 创建组件
    comp.start()  # 初始化
    # 初始化结束通知
    shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
    while not shared_memory[GlobalKey.ALL_READY.name]:
        time.sleep(0.1)
    comp.update()  # 算法逻辑循环