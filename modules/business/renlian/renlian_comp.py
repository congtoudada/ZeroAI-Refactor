import os
import time
import traceback
from typing import Dict
import cv2
import numpy as np
from loguru import logger

from business.count.count_comp import CountComponent
from business.renlian.renlian_info import RenlianInfo
from business.renlian.renlian_item import RenlianItem
from insight.zero.component.face_helper import FaceHelper
from zero.core.key.global_key import GlobalKey
from zero.utility.config_kit import ConfigKit
from zero.utility.img_kit import ImgKit
from zero.utility.object_pool import ObjectPool


class RenlianComponent(CountComponent):
    """
    计数的同时进行人脸识别
    """
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory, config_path)
        # 重新定义
        self.config: RenlianInfo = RenlianInfo(ConfigKit.load(config_path))
        self.pname = f"[ {os.getpid()}:renlian for {self.config.input_ports[0]}]"
        self.pool: ObjectPool = ObjectPool(20, RenlianItem)  # 对象池
        self.item_dict: Dict[int, RenlianItem] = {}  # 当前检测对象字典
        self.face_helper: FaceHelper = None

    def on_start(self):
        super().on_start()
        self.face_helper = FaceHelper(self.config.count_face_config, self.cam_id, self._face_callback)

    def on_update(self) -> bool:
        if super().on_update():
            for key, value in self.item_dict.items():
                if self._cen_send(key, value):
                    self.face_helper.send(key, self._crop_img(self.frames[0], value.ltrb))
                    # break  # 每帧最多发送一个请求（待定）
        self.face_helper.tick()
        return True

    def _cen_send(self, obj_id, item):
        diff = self.frame_id_cache[0] - item.last_send_req
        if self.face_helper.can_send(obj_id, diff, item.base_y, item.retry):
            self.item_dict[obj_id].last_send_req = self.frame_id_cache[0]
            return True
        else:
            return False

    def send_result(self, frame, status: int, item: RenlianItem):
        """
        结果通知
        :param status: 1进2出
        :param item:
        :return:
        """
        # 导出图
        time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        status_str = "In" if status == 1 else "Out"
        img_path = os.path.join(self.output_dir[0], f"{time_str}_{status_str}_{item.per_id}.jpg")
        img_shot = ImgKit.crop_img(frame, item.ltrb)
        if self.config.stream_export_img_enable:
            cv2.imwrite(img_path, img_shot)
            logger.info(f"{self.pname} 存图成功，路径: {img_path}")

        if self.config.stream_web_enable:
            # 通知后端
            data = {
                "recordTime": time_str,
                "camId": self.cam_id,
                "status": status,
                "personId": item.per_id,
                "shotImg": img_path
            }
            # WebKit.post(f"{WebKit.Prefix_url}/count", data)
            self.http_helper.post("algorithm/face", data)

    def on_destroy_obj(self, obj_id):
        self.face_helper.destroy_obj(obj_id)

    def on_create_obj(self, obj: RenlianItem):
        obj.last_face_req = self.frame_id_cache[0]

    def on_draw_vis(self, idx, frame, input_mot):
        frame = super().on_draw_vis(idx, frame, input_mot)
        face_dict = self.face_helper.face_dict
        # 参考线
        point1 = (0, int(self.face_helper.config.face_cull_up_y * self.stream_height))
        point2 = (self.stream_width, int(self.face_helper.config.face_cull_up_y * self.stream_height))
        point3 = (0, int((1 - self.face_helper.config.face_cull_down_y) * self.stream_height))
        point4 = (self.stream_width, int((1 - self.face_helper.config.face_cull_down_y) * self.stream_height))
        cv2.line(frame, point1, point2, (127, 127, 127), 1)  # 绘制线条
        cv2.line(frame, point3, point4, (127, 127, 127), 1)  # 绘制线条
        # 人脸识别结果
        for key, value in face_dict.items():
            ltrb = self.item_dict[key].ltrb
            cv2.putText(frame, f"{face_dict[key]['per_id']}",
                        (int((ltrb[0] + ltrb[2]) / 2), int(self.item_dict[key].ltrb[1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1)
        # 可视化并返回
        return frame

    def _face_callback(self, obj_id, per_id, score):
        if self.item_dict.__contains__(obj_id):
            if per_id == 1:
                self.item_dict[obj_id].retry += 1
            self.item_dict[obj_id].per_id = per_id
            self.item_dict[obj_id].score = score

    def _crop_img(self, im, ltrb):
        x1, y1, x2, y2 = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))
        # return im[int(y1): int(y2), int(x1): int(x2)]

    def _crop_img_border(self, im, ltrb, border=0):
        x1, y1, w, h = ltrb[0], ltrb[1], ltrb[2] - ltrb[0], ltrb[3] - ltrb[1]
        x2 = x1 + w + border
        x1 = x1 - border
        y2 = y1 + h + border
        y1 = y1 - border
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = im.shape[1] if x2 > im.shape[1] else x2
        y2 = im.shape[0] if y2 > im.shape[0] else y2
        return np.ascontiguousarray(np.copy(im[int(y1): int(y2), int(x1): int(x2)]))


def create_process(shared_memory, config_path: str):
    comp: RenlianComponent = RenlianComponent(shared_memory, config_path)  # 创建组件
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
        # 使用 traceback 打印堆栈信息
        traceback_info = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
        logger.error(f"RenlianComponent: {e}\n{traceback_info}")
        comp.on_destroy()


if __name__ == '__main__':
    arr1 = [0, 1]
    arr2 = [0]
    arr3 = np.array(arr1)
    arr4 = np.array(arr2)
    print(arr1 == arr2)
    print(np.array_equal(arr3, arr4))


    # [0,0.5] [0.75,0.5] [1, 0.6]
    # [0.25,0.25]
    ref_vec = np.array([0.75, 0, 0, -0.25, 0.1, 0]).reshape(2, 3)
    input_vec = np.array([0.25, -0.25, 0])
    print(np.dot(ref_vec, input_vec)[np.dot(ref_vec, input_vec) < 0])
    # print(np.cross(ref_vec, input_vec))

    # ref_vec = np.array([0.75, 0, 0])
    # ref_vec = np.array([0.25, 0.1, 0])
    # print(np.cross(ref_vec, input_vec))
