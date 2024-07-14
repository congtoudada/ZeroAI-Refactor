
class DetectionRecord:
    """
    用于临时缓存Detection结果
    """
    def __init__(self):
        self.ltrb = None
        self.score = 0
        self.cls = -1

    def init(self, ltrb, score, cls):
        self.ltrb = ltrb
        self.score = score
        self.cls = cls


