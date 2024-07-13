from zero.core.info.base_info import BaseInfo


class SimpleHttpInfo(BaseInfo):
    def __init__(self, data: dict = None):
        self.web_address = "localhost:8080"  # 请求地址前缀
        super().__init__(data)  # 前面是声明，一定要最后调用这段赋值
