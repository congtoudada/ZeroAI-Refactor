import multiprocessing
import os
import sys
import time

import cv2
from loguru import logger

from insight.zero.component.face_recognizer import FaceRecognizer
from insight.zero.info.insight_info import InsightInfo
from insight.zero.key.face_key import FaceKey
from zero.core.component.component import Component
from zero.core.helper.analysis_helper import AnalysisHelper
from zero.core.key.global_key import GlobalKey
from zero.utility.config_kit import ConfigKit
from zero.utility.timer_kit import TimerKit


class InsightComponent(Component):
    """
    Insight人脸识别服务:
        1.所有请求会发送到一个Req Queue，由Insight服务轮询处理。举例: Ultradict['FACE_REQ'].put({请求数据(含pid)})
        2.每个请求方需主动开辟一块共享内存作为Rsp Queue，Insight会把处理后的结果根据请求pid放到相应位置。举例: Ultradict['FACE_RSP'+pid].put({响应数据})
    """
    def __init__(self, shared_memory, config_path: str):
        super().__init__(shared_memory)
        self.config: InsightInfo = InsightInfo(ConfigKit.load(config_path))  # 配置文件内容
        self.pname = f"[ {os.getpid()}:insight_face ]"
        self.req_queue = None  # 人脸请求队列
        self.database_file = os.path.join(os.path.dirname(self.config.insight_database),
                                          "database-{}.json".format(self.config.insight_rec_feature))  # 人脸特征库配置文件路径
        self.face_model: FaceRecognizer = FaceRecognizer(self.config)  # 人脸识别模型（含人脸检测、人脸关键点检测、特征对齐、人脸识别）
        self.time_flag = 0  # 时间标识，用于检查是否重建特征库
        self.check_database_time = max(self.config.update_fps * 30, 1800)  # 半分钟检查一次
        self.database_timer = TimerKit()  # 特征数据库构建计时器
        self.infer_timer = TimerKit()  # 推理计时器

    def on_start(self):
        super().on_start()
        # 初始化请求缓存
        self.req_queue = multiprocessing.Manager().Queue()
        self.shared_memory[FaceKey.FACE_REQ.name] = self.req_queue

    def on_update(self) -> bool:
        # 检查特征库是否需要重建
        self.time_flag = (self.time_flag + 1) % sys.maxsize
        if self.time_flag % self.check_database_time == 0:
            if not os.path.exists(self.database_file):
                self.database_timer.tic()
                self.face_model.create_database(self.config.insight_database)
                self.database_timer.toc()
                # 记录构建特征库平均耗时
                if self.config.log_analysis:
                    AnalysisHelper.refresh("人脸特征库构建", f"{self.database_timer.average_time * 1000:.6f}ms", "无限制")

        # 处理请求
        while not self.req_queue.empty():
            self.infer_timer.tic()
            req = self.req_queue.get()
            cam_id = req[FaceKey.FACE_REQ_CAM_ID.name]  # 请求的摄像头id
            pid = req[FaceKey.FACE_REQ_PID.name]  # 请求的进程
            obj_id = req[FaceKey.FACE_REQ_OBJ_ID.name]  # 请求的对象id
            face_image = req[FaceKey.FACE_REQ_IMAGE.name]  # 请求的图片
            # 人脸识别处理
            per_id, score = self.face_model.search_face_image(face_image, self.config.insight_vis)
            # 响应输出结果
            rsp_key = FaceKey.FACE_RSP.name + str(pid)
            if self.shared_memory.__contains__(rsp_key):
                self.shared_memory[rsp_key].put({
                    FaceKey.FACE_RSP_OBJ_ID.name: obj_id,
                    FaceKey.FACE_RSP_PER_ID.name: per_id,
                    FaceKey.FACE_RSP_SCORE.name: score
                })
            self.infer_timer.toc()
            # break  # 每次最多处理一个响应
        # 记录推理平均耗时
        if self.config.log_analysis:
            AnalysisHelper.refresh("人脸推理", f"{self.infer_timer.average_time * 1000:.3f}ms", "500ms")

        return False

    def on_destroy(self):
        self.face_model.save()  # 保存数据库


def create_process(shared_memory, config_path: str):
    comp = InsightComponent(shared_memory, config_path)
    comp.start()
    shared_memory[GlobalKey.LAUNCH_COUNTER.name] += 1
    comp.update()


if __name__ == '__main__':
    img = cv2.imread('res/images/face/database/48-0001.jpg')
    config: InsightInfo = InsightInfo(ConfigKit.load("conf/dev/modules/face/insight/insight.yaml"))
    face_model: FaceRecognizer = FaceRecognizer(config)
    timerKit = TimerKit()
    print("开始预测")
    for i in range(10):
        timerKit.tic()
        per_id, score = face_model.search_face_image(img, config.insight_vis)
        print(f"{per_id} {score}")
        timerKit.toc()
    logger.info(f"耗时: {timerKit.average_time:.6f}")
