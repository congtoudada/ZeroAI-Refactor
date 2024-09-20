from zero.core.info.based_stream_info import BasedStreamInfo


class PhoneInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.detection_labels = []  # 手机检测类别
        self.phone_valid_count = 5  # 对象有效帧阈值
        self.phone_lost_frame = 60  # 消失多少帧丢弃
        self.phone_y_sort = False  # 分配id前进行y轴排序（时间换精度）
        self.phone_zone = []  # 手机检测区域 ltrb百分比 <-> [0][1][2][3]
        self.phone_timing_enable = True
        self.phone_warning_path = "output/business/phone/warning"
        self.phone_timing_path = "output/business/phone/timing"
        self.phone_timing_delta = 5
        self.phone_warning_uncropped_path = ""
        self.reid_uri = "/process2"
        self.reid_gallery_path = "res/images/reid_tmp_data/id_gt"
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
