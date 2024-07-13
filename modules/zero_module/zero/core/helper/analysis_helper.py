from UltraDict import UltraDict
from loguru import logger
from tabulate import tabulate


class AnalysisHelper:
    """
    性能分析器
    """
    global_shared_memory = UltraDict(name="analysis")

    @staticmethod
    def register_key(key):
        if not AnalysisHelper.global_shared_memory.__contains__(key):
            AnalysisHelper.global_shared_memory[key] = []

    @staticmethod
    def refresh(key, value, ref):
        """
        刷新性能数据
        :param key:
        :param value:
        :param ref: 参考值
        :return:
        """
        AnalysisHelper.register_key(key)
        AnalysisHelper.global_shared_memory[key] = [key, value, ref]

    @staticmethod
    def show():
        show_data = []
        for key in AnalysisHelper.global_shared_memory.keys():
            show_data.append(AnalysisHelper.global_shared_memory[key])
        headers = ["键", "值", "参考值"]
        table = tabulate(show_data, headers=headers, tablefmt="grid")
        logger.info("\n" + table)
        return table
