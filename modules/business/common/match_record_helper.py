import sys


class MatchRecordHelper:
    """
    为追踪结果匹配最优的其他类别检测框
    """
    @staticmethod
    def match_bbox(main_ltrb, sub_ltrbs, border = 0):
        """
        匹配第一个在main_ltrb内的sub_ltrb
        :param main_ltrb: (l, t, r, b)
        :param sub_ltrbs: list[(l, t, r, b)]
        :param border: main_ltrb扩张border，增大匹配成功率
        :return: sub_ltrbs下标, 如果全部sub不满足返回-1
        """
        for i, ltrb in enumerate(sub_ltrbs):
            if (ltrb[0] < main_ltrb[0] + border and ltrb[2] > main_ltrb[2] - border and
                    ltrb[1] < main_ltrb[1] + border and ltrb[3] > main_ltrb[3] - border):
                return i
        return -1

    @staticmethod
    def match_distance(main_ltrb, sub_ltrbs, max_distance=-1):
        """
        匹配离main_ltrb最近的sub_ltrb
        :param main_ltrb: (l, t, r, b)
        :param sub_ltrbs: list[(l, t, r, b)]
        :param max_distance: 最远距离，像素距离超过该值则剔除（-1表示无限制）
        :return: sub_ltrbs下标, 如果全部sub不满足返回-1
        """
        center = ((main_ltrb[0] + main_ltrb[2]) * 0.5, (main_ltrb[1] + main_ltrb[3]) * 0.5)
        min_dis = sys.maxsize
        min_idx = -1
        for i, ltrb in enumerate(sub_ltrbs):
            sub_center = ((sub_ltrbs[0] + sub_ltrbs[2]) * 0.5, (sub_ltrbs[1] + sub_ltrbs[3]) * 0.5)
            diff_x = (center[0] - sub_center[0]) * (center[0] - sub_center[0])
            diff_y = (center[1] - sub_center[1]) * (center[1] - sub_center[1])
            diff = diff_x + diff_y
            if (max_distance == -1 or diff <= max_distance) and diff < min_dis:
                min_dis = diff
                min_idx = i
        return min_idx