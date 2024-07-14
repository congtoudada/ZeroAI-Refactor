from zero.core.info.based_stream_info import BasedStreamInfo


class IntrudeInfo(BasedStreamInfo):
    def __init__(self, data: dict = None):
        self.intrude_valid_count = 5  # 对象有效报警帧阈值
        self.intrude_lost_frame = 60  # 消失多少帧丢弃
        self.intrude_zone = []  # 区域点
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
