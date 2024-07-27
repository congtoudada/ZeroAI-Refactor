from zero.core.info.base_info import BaseInfo


class SimpleHttpHelperInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.output_port = "SimpleHttp"  # http请求端口
        super().__init__(data)   # 前面是声明，一定要最后调用这段赋值
