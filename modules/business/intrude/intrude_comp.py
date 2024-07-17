import os
import time
from typing import Dict
import cv2
import numpy as np
from loguru import logger

from business.intrude.intrude_info import IntrudeInfo
from business.intrude.intrude_item import IntrudeItem
from bytetrack.zero.component.bytetrack_helper import BytetrackHelper
from zero.core.component.based_stream_comp import BasedStreamComponent
from zero.core.helper.warn_helper import WarnHelper
from zero.core.key.detection_key import DetectionKey
from zero.core.key.global_key import GlobalKey
from zero.core.key.stream_key import StreamKey
from zero.utility.config_kit import ConfigKit
from zero.utility.img_kit import ImgKit
from zero.utility.object_pool import ObjectPool


class IntrudeComponent(BasedStreamComponent):
    """
    特定区域入侵检测
    """
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: IntrudeInfo = IntrudeInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:intrude for {self.config.input_ports[0]}]"
        self.pool: ObjectPool = ObjectPool(20, IntrudeItem)
        self.cam_id = 0
        self.stream_width = 0
        self.stream_height = 0
        self.data_dict: Dict[int, IntrudeItem] = {}
        self.zone_points = []
        self.zone_vec = []
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 追踪器

    def on_start(self):
        super().on_start()
        self.cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]
        self.stream_width = int(self.read_dict[0][StreamKey.STREAM_WIDTH.name])
        self.stream_height = int(self.read_dict[0][StreamKey.STREAM_HEIGHT.name])
        # 预计算
        for point_str in self.config.intrude_zone:
            per_x = float(point_str.split(',')[0])
            per_y = float(point_str.split(',')[1])
            self.zone_points.append((per_x, per_y))
        for i in range(len(self.zone_points)):  # 最后一个点除外
            if i == 0:
                continue
            vec = (self.zone_points[i][0] - self.zone_points[i - 1][0],
                   self.zone_points[i][1] - self.zone_points[i - 1][1],
                   0)
            self.zone_vec.append(vec / np.linalg.norm(vec))

    def on_update(self) -> bool:
        self.release_unused()  # 清理无用资源（一定要在最前面调用）
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
        if input_det is None:
            return None
        mot_result = self.tracker.inference(input_det)  # 返回对齐输出后的mot结果
        # 根据mot结果进行计数
        self._intrude_core(frame, mot_result, self.frame_id_cache[0], frame.shape[1], frame.shape[0])
        return mot_result

    def _intrude_core(self, frame, input_mot, current_frame_id, width, height) -> bool:
        """
        # mot output shape: [n, 7]
        # n: n个对象
        # [0,1,2,3]: tlbr bboxes (基于视频流分辨率)
        #   [0]: x1
        #   [1]: y1
        #   [2]: x2
        #   [3]: y2
        # [4]: 置信度
        # [5]: 类别 (下标从0开始)
        # [6]: id
        """
        for obj in input_mot:
            ltrb = obj[:4]
            conf = obj[4]
            cls = int(obj[5])
            obj_id = int(obj[6])
            if cls == 0:  # 人
                # 更新Item状态
                if not self.data_dict.__contains__(obj_id):  # 没有被记录过
                    item = self.pool.pop()
                    item.init(obj_id, current_frame_id)
                    self.data_dict[obj_id] = item
                else:  # 已经记录过
                    in_warn = self._is_in_warn(ltrb)  # 判断是否处于警戒区
                    self.data_dict[obj_id].update(current_frame_id, in_warn)
                # 处理Item结果
                item = self.data_dict[obj_id]
                # 如果Item没有报过警且报警帧数超过有效帧，判定为入侵异常
                if not item.has_warn and item.get_valid_count() > self.config.intrude_valid_count:
                    logger.info(f"{self.pname} obj_id: {obj_id} 入侵异常")
                    shot_img = ImgKit.crop_img(frame, ltrb)
                    item.has_warn = True
                    WarnHelper.send_warn_result(self.pname, self.output_dir[0], self.cam_id,
                                                4, 1, shot_img,
                                                self.config.stream_export_img_enable,
                                                self.config.stream_web_enable)

    def release_unused(self):
        """
        清空长期未更新点
        :return:
        """
        clear_keys = []
        for key, item in self.data_dict.items():
            if self.frame_id_cache[0] - item.last_update_id > self.config.intrude_lost_frame:
                clear_keys.append(key)
        clear_keys.reverse()  # 从尾巴往前删除，确保索引正确性
        for key in clear_keys:
            self.pool.push(self.data_dict[key])
            self.data_dict.pop(key)  # 从字典中移除item

    def _is_in_warn(self, ltrb) -> bool:
        base_x, base_y = self.cal_center(ltrb)
        epsilon = 1e-3
        tmp = -1
        for i in range(len(self.zone_points) - 1):  # 最后一个点不计算
            p2o = (base_x - self.zone_points[i][0], base_y - self.zone_points[i][1], 0)  # 区域点->当前点的向量
            p2o_len = np.linalg.norm(p2o)
            if (abs(p2o_len)) > epsilon:  # 避免出现零向量导致叉乘无意义
                cross_z = np.cross(self.zone_vec[i], p2o)[2]  # (n, 1) n为red_vec
                if i == 0:
                    tmp = cross_z
                else:
                    if tmp * cross_z < 0:  # 出现异号说明不在区域内
                        return False
        return True

    def cal_center(self, ltrb):
        """
        计算中心点视口坐标作为2D参考坐标
        :param ltrb:
        :return:
        """
        center_x = (ltrb[0] + ltrb[2]) * 0.5 / self.stream_width
        center_y = (ltrb[1] + ltrb[3]) * 0.5 / self.stream_height
        return center_x, center_y

    def on_draw_vis(self, idx, frame, input_mot):
        text_scale = 1
        text_thickness = 1
        line_thickness = 2
        # 标题线
        num = 0 if input_mot is None else input_mot.shape[0]
        cv2.putText(frame, 'inference_fps:%.2f num:%d' %
                    (1. / max(1e-5, self.update_timer.average_time),
                     num), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 警戒线
        for i, point in enumerate(self.zone_points):
            if i == 0:
                continue
            cv2.line(frame, (int(self.zone_points[i][0] * self.stream_width), int(self.zone_points[i][1] * self.stream_height)),
                     (int(self.zone_points[i - 1][0] * self.stream_width),
                      int(self.zone_points[i - 1][1] * self.stream_height)),
                     (0, 0, 255), line_thickness)  # 绘制线条

        # 对象基准点、包围盒
        if input_mot is not None:
            for obj in input_mot:
                cls = obj[5]
                if cls == 0:
                    ltrb = obj[:4]
                    obj_id = int(obj[6])
                    screen_x = int((ltrb[0] + ltrb[2]) * 0.5)
                    screen_y = int((ltrb[1] + ltrb[3]) * 0.5)
                    cv2.circle(frame, (screen_x, screen_y), 4, (118, 154, 242), line_thickness)
                    cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                                  color=(0, 0, 255), thickness=line_thickness)
                    cv2.putText(frame, f"{obj_id}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=text_thickness)
                    if self.data_dict.__contains__(obj_id):
                        if self.data_dict[obj_id].has_warn:
                            cv2.putText(frame, "error",
                                        (int(ltrb[0] + 50), int(ltrb[1])),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=text_thickness)
                        else:
                            cv2.putText(frame, "normal",
                                        (int(ltrb[0] + 50), int(ltrb[1])),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=text_thickness)
        # 可视化并返回
        return frame


def create_process(shared_memory, config_path: str):
    comp: IntrudeComponent = IntrudeComponent(shared_memory, config_path)  # 创建组件
    try:
        comp.start()  # 初始化
        # 初始化结束通知
        shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
        while not shared_memory[GlobalKey.ALL_READY.name]:
            time.sleep(0.1)
        comp.update()  # 算法逻辑循环
    except KeyboardInterrupt:
        comp.on_destroy()
    except Exception as e:
        logger.error(f"IntrudeComponent: {e}")
        comp.on_destroy()
