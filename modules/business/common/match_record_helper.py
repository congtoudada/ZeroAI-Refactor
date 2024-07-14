import sys
from typing import List
from loguru import logger

from business.common.detection_record import DetectionRecord


class MatchRecordHelper:
    """
    为追踪结果匹配最优的其他类别检测框
    """
    @staticmethod
    def match_bbox(main_ltrb, record_list: List[DetectionRecord], border=0):
        """
        匹配第一个在main_ltrb内的sub_ltrb
        :param main_ltrb: (l, t, r, b)
        :param sub_ltrbs: list[(l, t, r, b)]
        :param border: main_ltrb扩张border，增大匹配成功率
        :return: sub_ltrbs下标, 如果全部sub不满足返回-1
        """
        for i, record in enumerate(record_list):
            if record.has_match:
                continue
            ltrb = record.ltrb
            center_x = (ltrb[0] + ltrb[2]) / 2.
            center_y = (ltrb[1] + ltrb[3]) / 2.
            if (main_ltrb[0] - border < center_x < main_ltrb[2] + border and
                    main_ltrb[1] - border < center_y < main_ltrb[3] + border):
                return i
        return -1

    @staticmethod
    def match_distance_l2(main_ltrb, record_list: List[DetectionRecord], max_distance=-1, cull_x=False, cull_y=False):
        """
        匹配离main_ltrb最近的sub_ltrb
        :param cull_y:
        :param cull_x:
        :param main_ltrb: (l, t, r, b)
        :param sub_ltrbs: list[(l, t, r, b)]
        :param max_distance: 最远距离，像素距离超过该值则剔除（-1表示无限制）
        :return: sub_ltrbs下标, 如果全部sub不满足返回-1
        """
        if cull_x and cull_y:
            logger.warning("同时剔除x和y，请检查代码逻辑！")
        center = ((main_ltrb[0] + main_ltrb[2]) * 0.5, (main_ltrb[1] + main_ltrb[3]) * 0.5)
        min_dis = sys.maxsize
        min_idx = -1
        for i, record in enumerate(record_list):
            if record.has_match:
                continue
            ltrb = record.ltrb
            sub_center = ((ltrb[0] + ltrb[2]) * 0.5, (ltrb[1] + ltrb[3]) * 0.5)
            diff_x = 0
            diff_y = 0
            if not cull_x:
                diff_x = (center[0] - sub_center[0]) * (center[0] - sub_center[0])
            if not cull_y:
                diff_y = (center[1] - sub_center[1]) * (center[1] - sub_center[1])
            diff = diff_x + diff_y
            if (max_distance == -1 or diff <= max_distance * max_distance) and diff < min_dis:
                min_dis = diff
                min_idx = i
        return min_idx

    @staticmethod
    def match_bbox_old(main_ltrb, sub_ltrbs, border=0):
        """
        匹配第一个在main_ltrb内的sub_ltrb
        :param main_ltrb: (l, t, r, b)
        :param sub_ltrbs: list[(l, t, r, b)]
        :param border: main_ltrb扩张border，增大匹配成功率
        :return: sub_ltrbs下标, 如果全部sub不满足返回-1
        """
        for i, ltrb in enumerate(sub_ltrbs):
            center_x = (ltrb[0] + ltrb[2]) / 2.
            center_y = (ltrb[1] + ltrb[3]) / 2.
            if (main_ltrb[0] - border < center_x < main_ltrb[2] + border and
                    main_ltrb[1] - border < center_y < main_ltrb[3] + border):
                return i
        return -1


    @staticmethod
    def match_distance_l2_old(main_ltrb, sub_ltrbs, max_distance=-1, cull_x=False, cull_y=False):
        """
        匹配离main_ltrb最近的sub_ltrb
        :param cull_y:
        :param cull_x:
        :param main_ltrb: (l, t, r, b)
        :param sub_ltrbs: list[(l, t, r, b)]
        :param max_distance: 最远距离，像素距离超过该值则剔除（-1表示无限制）
        :return: sub_ltrbs下标, 如果全部sub不满足返回-1
        """
        if cull_x and cull_y:
            logger.warning("同时剔除x和y，请检查代码逻辑！")
        center = ((main_ltrb[0] + main_ltrb[2]) * 0.5, (main_ltrb[1] + main_ltrb[3]) * 0.5)
        min_dis = sys.maxsize
        min_idx = -1
        for i, ltrb in enumerate(sub_ltrbs):
            sub_center = ((ltrb[0] + ltrb[2]) * 0.5, (ltrb[1] + ltrb[3]) * 0.5)
            diff_x = 0
            diff_y = 0
            if not cull_x:
                diff_x = (center[0] - sub_center[0]) * (center[0] - sub_center[0])
            if not cull_y:
                diff_y = (center[1] - sub_center[1]) * (center[1] - sub_center[1])
            diff = diff_x + diff_y
            if (max_distance == -1 or diff <= max_distance * max_distance) and diff < min_dis:
                min_dis = diff
                min_idx = i
        return min_idx
