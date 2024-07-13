import argparse

from zero.core.component.launch_comp import LaunchComponent


def make_parser():
    parser = argparse.ArgumentParser("ZeroAI Demo!")
    parser.add_argument("-app", "--application", type=str, default="conf/application-dev.yaml")
    return parser


if __name__ == '__main__':
    # 解析args
    args = make_parser().parse_args()
    comp = LaunchComponent(args.application)
    comp.start()
    comp.update()

