from prefect import Task
from loguru import logger
from dynaconf import settings
import json
import random
from transformers import AutoTokenizer


class PreparePixelHelpPair(Task):
    def run(self, file_location):
        """Parses the RicoSCA dataset for the pair classification task.
        Args:
            file_location (str): location of the RicoSCA dataset

        Returns:
            Dict: preprocessed dict in the following format:
            
            {
            "id_query": id of the query (instruction)    
            instruction: NL intruction,
            ui: DICT:
                    text: text of the ui element,
                    x0: bounding box x0,
                    x1: bouding box x1,
                    y0: bounding box y0,
                    y1: bounding box y1,
            label: 1 if UI element is refered in the instruction, 0 otherwise
            }

            mapping_query: Dict:
            {query_id: expected_selected_ui_element
            }

        """

        total_pairs = 0
        total_negative_pairs = 0
        total_positive_pairs = 0
        index_query = 0
        total_ui_elements = 0
        largest_text = 0

        parsed_data = dict()

        logger.info("Preprocessing PixelHELP dataset")
        with open(file_location, "r") as f:
            input_data = json.load(f)

        mapping_query = dict()
        for _, screen_info in input_data.items():
            selected_ui_element = screen_info["labels"]
            mapping_query[index_query] = int(selected_ui_element)
            for ui_index, ui_element in screen_info["ui_elements"].items():

                if int(ui_index) == selected_ui_element:
                    label_ui = 1
                    total_positive_pairs = total_positive_pairs + 1
                else:
                    total_negative_pairs = total_negative_pairs + 1
                    label_ui = 0

                parsed_data[total_pairs] = {
                    "id_query": index_query,
                    "ui_position": int(ui_index),
                    "instruction": screen_info["instruction"],
                    "ui": ui_element,
                    "label": label_ui,
                }
                total_pairs = total_pairs + 1
            index_query = index_query + 1

        logger.info(f"******** LARGEST UI TEXT: {largest_text} ********")
        logger.info(f"***** TOTAL UI ELEMENTS: {total_ui_elements}")
        logger.info(f"Total of pairs: {total_pairs}")
        logger.info(f"Total negative pairs: {total_negative_pairs}")
        logger.info(f"Total positive pairs: {total_positive_pairs}")

        return {"data": parsed_data, "mapping": mapping_query}
