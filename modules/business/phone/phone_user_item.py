
class PhoneUserItem:
    def __init__(self):
        self.obj_id = 0  # 目标id
        self.last_update_id = 0  # 上次更新帧
        self.valid_count = 0  # 有效更新阈值（连续检测多少帧报警，才报警）
        self.cls: int = 1  # 0: phone 其他: 干扰项
        self.has_warn = False  # 是否已经报警
        self.warn_score = 0  # 报警置信度
        self.ltrb = (0, 0, 1, 1)  # 人的包围框
        self.phone_ltrb = (0, 0, 1, 1)  # 手机的包围框
        self.has_match = False  # 当前帧是否成功匹配

    def init(self, obj_id, cls, score, last_update_id):
        self.obj_id = obj_id
        self.last_update_id = last_update_id
        self.valid_count = 0
        self.cls = cls
        self.has_warn = False
        self.warn_score = score  # 报警置信度
        self.ltrb = (0, 0, 1, 1)  # 包围框
        self.phone_ltrb = (0, 0, 1, 1)  # 手机的包围框
        self.has_match = False  # 当前帧是否成功匹配

    def match_update(self, person_ltrb, cls, score, phone_ltrb):
        self.has_match = True
        self.ltrb = person_ltrb
        self.warn_score = score
        self.phone_ltrb = phone_ltrb
        if not self.has_warn:
            if self.cls == cls:
                self.valid_count += 2
            else:
                self.valid_count = 0
                self.cls = cls

    def late_update(self, last_update_id):
        self.last_update_id = last_update_id
        if not self.has_match:  # 当前帧未匹配上，扣分
            self.has_match = False
            self.valid_count -= 1
            if self.valid_count < 0:
                self.valid_count = 0

    def get_valid_count(self):
        return self.valid_count

