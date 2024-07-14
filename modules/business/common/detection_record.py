
class DetectionRecord:
    """
    用于临时缓存Detection结果
    """
    def __init__(self):
        self.ltrb = None
        self.score = 0
        self.cls = -1
        self.has_match = False  # 是否已匹配
        self.match_id = -1  # 匹配的对象id，用于debug调试

    def init(self, ltrb, score, cls):
        self.ltrb = ltrb
        self.score = score
        self.cls = cls
        self.has_match = False  # 是否已匹配
        self.match_id = -1  # 匹配的对象id，用于debug调试

    def match_action(self, obj_id):
        self.has_match = True
        self.match_id = obj_id


