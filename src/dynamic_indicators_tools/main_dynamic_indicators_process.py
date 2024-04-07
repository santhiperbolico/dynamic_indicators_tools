import json
import logging
import os
from datetime import datetime
from typing import List, Union

from dynamic_indicators_tools.dynamic_indicators.dynamic_indicators_utils import main_process_di


def multi_process_dynamic_indicators(config_json_path: Union[str, List[str]]) -> None:
    if not os.path.exists("logs"):
        os.mkdir("logs")
    path_log = os.path.join("logs", f"log_{datetime.today().strftime('%Y%m%d_%H%M')}.txt")
    logging.basicConfig(
        filename=path_log, level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s"
    )

    if isinstance(config_json_path, str):
        config_json_path = [config_json_path]

    counter_process = 0
    for config_json in config_json_path:
        with open(config_json, "r") as file_json:
            params = json.load(file_json)
        system_name = params.get("system_params").get("system_name")
        logging.info(f"Ejecutando archivo {system_name} - {counter_process}")
        main_process_di(params)
        counter_process += 1
