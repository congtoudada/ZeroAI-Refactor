import glob
import os
import time
import traceback
from typing import Dict, List
import cv2
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

from business.common.detection_record import DetectionRecord
from business.common.match_record_helper import MatchRecordHelper
from business.phone.phone_info import PhoneInfo
from business.phone.phone_user_item import PhoneUserItem
from bytetrack.zero.component.bytetrack_helper import BytetrackHelper
from simple_http.simple_http_helper import SimpleHttpHelper
from zero.core.component.based_stream_comp import BasedStreamComponent
from zero.core.helper.warn_helper import WarnHelper
from zero.core.key.detection_key import DetectionKey
from zero.core.key.global_key import GlobalKey
from zero.core.key.stream_key import StreamKey
from zero.utility.config_kit import ConfigKit
from zero.utility.img_kit import ImgKit_img_box
from zero.utility.object_pool import ObjectPool
import requests


class PhoneComponent(BasedStreamComponent):
    """
    手机检测组件
    """

    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: PhoneInfo = PhoneInfo(ConfigKit.load(config_path))  # 配置信息
        self.pname = f"[ {os.getpid()}:phone for {self.config.input_ports[0]}&{self.config.input_ports[1]} ]"  # 进程唯一标识
        self.cam_id = 0  # 对应的摄像头id
        self.stream_width = 0  # 画面宽
        self.stream_height = 0  # 画面高
        # key: obj_id value: cls
        self.user_pool: ObjectPool = ObjectPool(20, PhoneUserItem)  # PhoneUserItem对象池
        self.record_pool: ObjectPool = ObjectPool(20, DetectionRecord)  # 检测记录对象池
        self.user_dict: Dict[int, PhoneUserItem] = {}  # PhoneUser字典
        self.tracker: BytetrackHelper = BytetrackHelper(self.config.stream_mot_config)  # 人的追踪器
        self.phone_records: List[DetectionRecord] = []  # 手机目标检测结果（每帧更新）
        self.current_mot = None  # 当前帧人的追踪结果，如果非None，则最后要消耗掉检测结果
        self.timing_record1 = float('-inf')  # reid相关
        self.timing_record2 = float('-inf')  # reid相关
        self.warn_person_bboxes = []  # 报警时人的包围框
        self.warn_phone_bboxes = []  # 报警时手机的包围框
        self.warn_phone_score = []  # 报警时手机的置信度
        self.http_helper = SimpleHttpHelper(self.config.stream_http_config)  # http帮助类

    def on_start(self):
        """
        初始化时执行
        :return:
        """
        super().on_start()
        self.cam_id = self.read_dict[0][StreamKey.STREAM_CAM_ID.name]
        self.stream_width = self.read_dict[0][StreamKey.STREAM_WIDTH.name]
        self.stream_height = self.read_dict[0][StreamKey.STREAM_HEIGHT.name]

    def on_update(self) -> bool:
        self.release_unused()  # 清理无用资源（一定要在最前面调用）
        super().on_update()  # 核心更新逻辑
        # 后处理，如果存在人，则清空当前帧的检测结果
        if self.current_mot is not None:
            self.phone_records.clear()
            self.current_mot = None
        return True

    def on_resolve_per_stream(self, read_idx):
        """
        解析每个流的数据（帧、帧id，目标检测结果）
        :param read_idx:
        :return:
        """
        frame, _ = super().on_resolve_per_stream(read_idx)  # 解析视频帧id+视频帧
        if frame is None:  # 没有有效帧
            return frame, None
        # 解析额外数据
        stream_package = self.read_dict[read_idx][self.config.input_ports[read_idx]]
        input_det = stream_package[DetectionKey.DET_PACKAGE_RESULT.name]  # 目标检测结果
        return frame, input_det

    def on_process_per_stream(self, idx, frame, input_det):
        """
        处理每个流的数据
        :param idx: 从input_ports[idx]取package
        :param frame: 帧
        :param input_det: 目标检测结果
        :return:
        """
        if input_det is None:
            return None

        if idx == 0:  # 0号端口取的数据是手机检测结果
            for i in range(len(self.phone_records)):
                self.record_pool.push(self.phone_records[i])
            self.phone_records.clear()  # 清空手机检测记录
            for i, item in enumerate(input_det):  # 填充新的检测记录
                ltrb = (item[0], item[1], item[2], item[3])
                score = item[4]
                cls = item[5]
                record = self.record_pool.pop()
                record.init(ltrb, score, cls)
                self.phone_records.append(record)
            return None
        else:  # 1号端口取的是人的检测结果
            input_det = input_det[input_det[:, 5] == 0]
            mot_result = self.tracker.inference(input_det)  # 返回对齐输出后的mot结果
            self.current_mot = mot_result  # 缓存追踪结果（主要用于帧结束时判断是否消耗掉检测结果）
            # 定期存图
            if mot_result is not None:
                person_all_bboxes = []
                for i, obj in enumerate(mot_result):
                    ltrb = obj[:4]
                    person_all_bboxes.append(ltrb)
                self.check_and_save_timing_images(frame, person_all_bboxes)

            # 定期删除：每delta2秒执行一次，删去timing文件夹中已经存放了超过age_limit秒的旧图片
            delta2 = 180  # 定期清空历史抓拍图片，每delta2时间执行一次以免影响性能
            now = time.time()
            # print("调试123     ——————————",self.config.phone_timing_path)
            if (now - self.timing_record2) >= delta2:
                self.clear_old_files(self.config.phone_timing_path)  # output/business/phone/timing
                self.timing_record2 = now

            # 根据mot结果进行手机核心业务！！！
            self._phone_core(frame, mot_result, self.frame_id_cache[0], frame.shape[1], frame.shape[0])

            # 报警存图
            # print("tiaoshi+++++120", len(self.warn_person_bboxes))
            if len(self.warn_person_bboxes) > 0:
                self.save_warning_images(frame, self.warn_person_bboxes)
                # 交给reid模块处理并报警给后端
                print("——————————尝试发送手机报警请求给reid, 等待计算结果——————————————", self.warn_person_bboxes)
                data = {
                    "query_directory": self.config.phone_warning_path,  # 新增报警
                    "gallery_directory": self.config.reid_gallery_path,
                }
                # response = requests.post(self.config.reid_uri, json=data, headers={"Content-Type": "application/json"})  # 发送POST请求
                self.http_helper.post(uri=self.config.reid_uri, data=data)  # (异步!!)🟥
                self.warn_person_bboxes.clear()
                self.warn_phone_bboxes.clear()
                self.warn_phone_score.clear()

            return mot_result

    def _phone_core(self, frame, input_mot, current_frame_id, width, height) -> bool:
        if input_mot is None:
            return
        # 时间换精度: 根据t排序（y轴排序），配合包围盒匹配，可以使精度更高（默认关闭）
        if self.config.phone_y_sort:
            sort_indices = np.argsort(input_mot[:, 1])
            input_mot = input_mot[sort_indices]
            self.phone_records.sort(key=lambda x: x.ltrb[1])

        # 遍历每台手机，找到离最近的人
        for i, phone in enumerate(self.phone_records):
            phone_ltrb = phone.ltrb
            phone_base = (phone_ltrb[0] + phone_ltrb[2]) * 0.5, (phone_ltrb[1] + phone_ltrb[3]) * 0.5
            # 只有手机在检测区域内才匹配(通过l2距离计算结果)
            if not self._is_in_zone(phone_ltrb, self.config.phone_zone):
                continue

            min_id = -1  # 最近的人的索引(对应input_mot)
            min_dis = float("inf")
            for j, obj in enumerate(input_mot):
                ltrb = obj[:4]
                w = ltrb[2] - ltrb[0]
                h = ltrb[3] - ltrb[1]
                # 基准点计算为上半身中心
                upper_ltrb = ltrb + (0, 0, 0, -h / 2)
                upper_base = ((upper_ltrb[0] + upper_ltrb[2]) * 0.5, (upper_ltrb[1] + upper_ltrb[3]) * 0.5)
                # 计算距离
                dis = ((phone_base[0] - upper_base[0]) * (phone_base[0] - upper_base[0]) *
                       (phone_base[1] - upper_base[1]) * (phone_base[1] - upper_base[1]))
                if dis < min_dis:
                    min_id = j
                    min_dis = dis
            # 存在匹配项，更新人的信息
            if min_id != -1:
                obj = input_mot[min_id]
                obj_ltrb = obj[:4]
                obj_id = obj[6]
                # 人没有被记录过，则记录
                if not self.user_dict.__contains__(obj_id):
                    item = self.user_pool.pop()
                    item.init(obj_id, phone.cls, phone.score, current_frame_id)
                    self.user_dict[obj_id] = item
                # 成功匹配且已经记录过，更新人的状态
                self.user_dict[obj_id].match_update(obj_ltrb, phone.cls, phone.score, phone_ltrb)
        # 遍历一次人，汇总结果
        for i, obj in enumerate(input_mot):
            obj_id = obj[6]
            if not self.user_dict.__contains__(obj_id):
                continue
            # 计算结果，满足异常条件就记录
            self.process_result(frame, self.user_dict[obj_id])
            # 当前帧收尾
            self.user_dict[obj_id].late_update(current_frame_id)

    def process_result(self, frame, phone_item: PhoneUserItem):
        # 没有报过警且异常状态保持一段时间才发送
        if not phone_item.has_warn and phone_item.get_valid_count() >= self.config.phone_valid_count:
            if phone_item.cls == 0:  # 持有手机，报警！
                logger.info(
                    f"手机检测异常: obj_id:{phone_item.obj_id} warn_score:{phone_item.warn_score}")
                phone_item.has_warn = True  # 一旦视为异常，则一直为异常，避免一个人重复报警
                self.warn_person_bboxes.append(phone_item.ltrb)  # 报警人的包围框
                self.warn_phone_bboxes.append(phone_item.phone_ltrb)  # 报警手机的包围框
                self.warn_phone_score.append(phone_item.warn_score)  # 报警手机的分数
                # shot_img = ImgKit_img_box.draw_img_box(frame, ltrb)  # 画线，后续展示完整的图
                # self.http_helper.send_warn_result(self.pname, self.output_dir[0], self.cam_id, 1, 1,
                #                             shot_img, self.config.stream_export_img_enable,
                #                             self.config.stream_web_enable)
                # 如果发送异步post就直接调用self.post

    def release_unused(self):
        # 清空长期未更新点
        clear_keys = []
        for key, item in self.user_dict.items():
            if self.frame_id_cache[0] - item.last_update_id > self.config.phone_lost_frame:
                clear_keys.append(key)
        clear_keys.reverse()
        for key in clear_keys:
            self.user_pool.push(self.user_dict[key])
            self.user_dict.pop(key)  # 从字典中移除item

    def on_draw_vis(self, idx, frame, input_mot):
        """
        可视化
        :param idx:
        :param frame:
        :param input_mot:
        :return:
        """
        if input_mot is None:
            return None
        text_scale = 2
        text_thickness = 2
        line_thickness = 2
        # 标题线
        num = 0 if input_mot is None else input_mot.shape[0]
        cv2.putText(frame, 'inference_fps:%.2f num:%d' %
                    (1. / max(1e-5, self.update_timer.average_time),
                     num), (0, int(15 * text_scale)),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
        # 手机区域
        if len(self.config.phone_zone) > 0:
            phone_zone = self.config.phone_zone
            cv2.rectangle(frame, pt1=(int(phone_zone[0] * self.stream_width), int(phone_zone[1] * self.stream_height)),
                          pt2=(int(phone_zone[2] * self.stream_width), int(phone_zone[3] * self.stream_height)),
                          color=(0, 255, 0), thickness=line_thickness)
        # 对象基准点、包围盒
        if len(self.config.detection_labels) == 0:
            logger.warning(f"{self.pname} detection_labels的长度为0，请在配置文件中配置detection_labels!")
            return frame
        # 人
        if input_mot is not None:
            for obj in input_mot:
                ltrb = obj[:4]
                obj_id = int(obj[6])
                # screen_x = int((ltrb[0] + ltrb[2]) / 2)
                # screen_y = int((ltrb[1] + ltrb[3]) / 2)
                cv2.circle(frame, (int(ltrb[0]), int(ltrb[1])), 4, (118, 154, 242), line_thickness)
                cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                              color=self._get_color(obj_id), thickness=line_thickness)
                if self.user_dict.__contains__(obj_id):
                    cls = int(self.user_dict[obj_id].cls)
                    is_warn = self.user_dict[obj_id].has_warn
                    cv2.putText(frame, f"{obj_id}:{self.config.detection_labels[cls]} warn:{is_warn}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, self._get_color(obj_id), thickness=text_thickness)
                else:
                    cv2.putText(frame, f"{obj_id}",
                                (int(ltrb[0]), int(ltrb[1])),
                                cv2.FONT_HERSHEY_PLAIN, text_scale, self._get_color(obj_id), thickness=text_thickness)
        # 手机
        for i, item in enumerate(self.phone_records):
            ltrb = item.ltrb
            cls = int(item.cls)
            score = item.score
            cv2.rectangle(frame, pt1=(int(ltrb[0]), int(ltrb[1])), pt2=(int(ltrb[2]), int(ltrb[3])),
                          color=(0, 0, 255), thickness=line_thickness)
            id_text = f"obj:{item.match_id} {self.config.detection_labels[cls]}({score:.2f})"
            cv2.putText(frame, id_text, (int(ltrb[0]), int(ltrb[1])), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, (0, 0, 255), thickness=text_thickness)
        # 可视化并返回
        return frame

    def _get_color(self, idx):
        idx = (1 + idx) * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color

    def _is_in_zone(self, person_ltrb, phone_ltrb):
        if len(phone_ltrb) == 0:
            return True
        # base_x = ((person_ltrb[0] + person_ltrb[2]) / 2) / self.stream_width
        # base_y = ((person_ltrb[1] + person_ltrb[3]) / 2) / self.stream_height
        base_x = person_ltrb[0] / self.stream_width
        base_y = person_ltrb[1] / self.stream_height
        if phone_ltrb[0] < base_x < phone_ltrb[2] and phone_ltrb[1] < base_y < phone_ltrb[3]:
            return True
        else:
            return False

    def on_save_img_full(self, idx, img, bbox=None, path='.', draw_box=False, warn_score=None, phone_bbox=None):
        if not os.path.exists(path):
            os.makedirs(path)

        if img.size == 0:
            print("警告: 裁剪的图像为空。")
            return None, None

        if draw_box:  # 确保bbox不为空
            # if bbox is not None:
            #     img = self.draw_img_box(img, bbox)
            if phone_bbox is not None:
                img = self.draw_img_box(img, phone_bbox, 'blue')

        time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        # 在文件名中加入warn_score，确保warn_score为字符串形式
        image_name = f"0_{self.cam_id}_{time_str}_{idx}_{warn_score}.jpg" if warn_score is not None else f"0_{self.cam_id}_{time_str}_{idx}.jpg"
        image_path = os.path.join(path, image_name)

        try:
            cv2.imwrite(image_path, img)
        except Exception as e:
            print(f"错误: 保存图像失败 - {e}")
            return None, None

        return image_path, img

    def on_save_img_crop(self, idx, img, bbox=None, path='.', draw_box=False, warn_score=None):
        if not os.path.exists(path):
            os.makedirs(path)

        if draw_box:  # True
            print("警告: 保存逻辑错误。")
        if not draw_box:
            img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if img.size == 0:
            print("警告: 裁剪的图像为空。")
            return None, None

        time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        # 在文件名中加入warn_score，确保warn_score为字符串形式
        image_name = f"0_{self.cam_id}_{time_str}_{idx}_{warn_score}.jpg" if warn_score is not None else f"0_{self.cam_id}_{time_str}_{idx}.jpg"
        image_path = os.path.join(path, image_name)

        try:
            cv2.imwrite(image_path, img)
        except Exception as e:
            print(f"错误: 保存图像失败 - {e}")
            return None, None

        return image_path, img

    def draw_img_box(self, im, ltrb, color='red'):
        x1, y1, x2, y2 = ltrb
        im_pil = Image.fromarray(im)
        draw = ImageDraw.Draw(im_pil)
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=5)
        im_with_rectangle = np.array(im_pil)
        return im_with_rectangle

    def count_images_in_directory(self, directory):
        # 返回目录中的jpg图片数量
        return len(glob.glob(os.path.join(directory, '*.jpg')))

    def clear_directory(self, directory):
        # 删除目录中的所有文件
        files = glob.glob(os.path.join(directory, '*'))
        for f in files:
            os.remove(f)

    def check_and_save_timing_images(self, frame, all_bboxes):
        if frame is None:
            return
        # 检查是否启用了定时保存
        if not self.config.phone_timing_enable:
            print("调试：定时保存功能未启用！")
            return  # 如果未启用，则直接返回

        delta1 = self.config.phone_timing_delta  # 定期存图间隔为delta
        now = time.time()
        # 每delta1秒执行一次，存图操作
        if all_bboxes is not None:
            if (now - self.timing_record1) >= delta1:
                # if isinstance(all_bboxes, list):
                for i, bbox in enumerate(all_bboxes):
                    # print("调试374",self.config.phone_timing_path)
                    self.on_save_img_crop(i, frame, bbox, self.config.phone_timing_path)
                self.timing_record1 = now  # 记录上次存图时间

    def save_warning_images(self, frame, object_bboxes):
        if frame is None:
            return
        for i, bbox in enumerate(object_bboxes):
            # 保存一张完整warning图，并画上框
            self.on_save_img_full(i, img=frame, bbox=bbox, path=self.config.phone_warning_uncropped_path, draw_box=True,
                                  warn_score=f"{self.warn_phone_score[i]:.3f}", phone_bbox=self.warn_phone_bboxes[i])
            # 保存一张裁剪warning图
            self.on_save_img_crop(i, img=frame, bbox=bbox, path=self.config.phone_warning_path, draw_box=False,
                                  warn_score=f"{self.warn_phone_score[i]:.3f}")

    def clear_old_files(self, directory, age_limit=180):
        """
        清除指定目录下超过age_limit秒未修改的所有文件。
        :param directory: 要清理的目录路径
        :param age_limit: 文件保留的最大秒数，默认180秒（3分钟）
        """
        # 获取当前时间
        now = time.time()
        # print("调试393   clear_old_files")
        # 遍历目录中的所有文件
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            # 获取文件状态信息
            file_stat = os.stat(file_path)
            # 计算文件最后修改时间与当前时间的差值
            time_diff = now - file_stat.st_mtime

            # 如果文件修改时间超过了时间限制，则删除文件
            if time_diff > age_limit:
                # print(f"删除文件: {file_path}")
                os.remove(file_path)
                print(f"已删除超过{age_limit}秒未修改的文件: {filename}")


def create_process(shared_memory, config_path: str):
    comp: PhoneComponent = PhoneComponent(shared_memory, config_path)  # 创建组件
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
        logger.error(f"PhoneComponent: {e}\n{traceback_info}")
        comp.on_destroy()
