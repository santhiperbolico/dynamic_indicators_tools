import logging
from typing import List, Union

from dynamic_indicators_tools.config_files.get_params_config import get_main_params
from dynamic_indicators_tools.dynamic_indicators.dynamic_indicators_process import main_process_di


def multi_process_dynamic_indicators(config_json_path: Union[str, List[str]]) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s")

    if isinstance(config_json_path, str):
        config_json_path = [config_json_path]

    counter_process = 0
    for config_json in config_json_path:
        system, system_params, dynamic_indicators = get_main_params(config_json)
        logging.info(f"Ejecutando archivo {system_params['system_name']} - {counter_process}")
        main_process_di(system, system_params, dynamic_indicators)
        counter_process += 1
