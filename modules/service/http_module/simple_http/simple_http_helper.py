from UltraDict import UltraDict
from loguru import logger
from simple_http.simple_http_comp import SimpleHttpComponent
from simple_http.simple_http_key import SimpleHttpKey


class SimpleHttpHelper:
    shared_memory = UltraDict(name=SimpleHttpComponent.SHARED_MEMORY_NAME)

    @staticmethod
    def get(uri: str):
        req_package = {
            SimpleHttpKey.HTTP_PACKAGE_URI.name: uri,
            SimpleHttpKey.HTTP_PACKAGE_METHOD.name: 1,
            SimpleHttpKey.HTTP_PACKAGE_JSON.name: None
        }
        if not SimpleHttpHelper.shared_memory.__contains__(SimpleHttpKey.HTTP_REQ.name):
            logger.error("发送Http请求失败, 没有开启SimpleHttpService!")
            return
        SimpleHttpHelper.shared_memory[SimpleHttpKey.HTTP_REQ.name].put(req_package)

    @staticmethod
    def post(uri: str, data: dict):
        req_package = {
            SimpleHttpKey.HTTP_PACKAGE_URI.name: uri,
            SimpleHttpKey.HTTP_PACKAGE_METHOD.name: 2,
            SimpleHttpKey.HTTP_PACKAGE_JSON.name: data
        }
        if not SimpleHttpHelper.shared_memory.__contains__(SimpleHttpKey.HTTP_REQ.name):
            logger.error("发送Http请求失败, 没有开启SimpleHttpService!")
            return
        SimpleHttpHelper.shared_memory[SimpleHttpKey.HTTP_REQ.name].put(req_package)

