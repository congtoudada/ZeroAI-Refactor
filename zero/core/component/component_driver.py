from zero.core.component.component import Component
from loguru import logger


class ComponentDriver:
    @staticmethod
    def execute(root: Component):
        if root is None:
            logger.error("[ Fatal Error ] ComponentDriver's root is None, execute failed!")
            return
        # 组件初始化
        root.on_start()  # 先初始化父组件（通常在该函数内添加相应子组件）
        has_child = False  # 判断是否有子组件
        for child in root.children:  # 再执行子组件更新
            has_child = True
            child.on_start()
        # 组件更新
        while True:
            if root.enable:
                root.on_update()  # 先执行父组件的更新
                if has_child:
                    for child in root.children:  # 再执行子组件更新
                        if child.enable:
                            child.on_update()
            if root.esc_event.is_set():
                if has_child:
                    for child in root.children:  # 先销毁子组件
                        child.on_destroy()
                root.on_destroy()  # 再销毁父组件
                return


