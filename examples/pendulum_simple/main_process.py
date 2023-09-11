import os

from dynamic_indicators_tools.main_dynamic_indicators_process import (
    multi_process_dynamic_indicators,
)

PATH = "examples/pendulum_simple/config_files/"

CONFIG_FILES = [
    os.path.join(PATH, "config_main_pendulum_system_t_10.json"),
    os.path.join(PATH, "config_main_pendulum_system_t_30.json"),
]

if __name__ == "__main__":
    multi_process_dynamic_indicators(CONFIG_FILES)
