from business.count.count_info import CountInfo


class RenlianInfo(CountInfo):
    def __init__(self, data: dict = None):
        self.count_face_config = ""  # 人脸识别配置文件，发送人脸识别请求相关
        self.reid_output_path = "ZeroAI-Refactor/res/images/reid_tmp_data/id_gt"  # 存人的图片
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
