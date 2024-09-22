
class PhoneItem:
    def __init__(self):
        self.obj_id = 0  # 目标id
        self.last_update_id = 0  # 上次更新帧
        self.valid_count = 0  # 有效更新阈值（连续检测多少帧报警，才报警）
        self.cls: int = 1  # 0: phone 其他: 干扰项
        self.has_warn = False  # 是否已经报警
        self.warn_score = 0  # 报警置信度

    def init(self, obj_id, cls, score, last_update_id):
        self.obj_id = obj_id
        self.last_update_id = last_update_id
        self.valid_count = 0
        self.cls = cls
        self.has_warn = False
        self.warn_score = score  # 报警置信度

    def update(self, last_update_id, cls, score):
        self.last_update_id = last_update_id
        self.warn_score = score
        if not self.has_warn:
            if cls == self.cls:  # 当前检测类别和记录的类别相同，递增
                self.valid_count += 1
            else:  # 不同则重置
                self.valid_count = 0
                self.cls = cls

    def get_valid_count(self):
        return self.valid_count

