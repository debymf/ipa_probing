from prefect import Task
from loguru import logger
from dynaconf import settings
import json
import random
from transformers import AutoTokenizer
import math

TOTAL_SELECTED = 15


class PreparePixelHelpBasicPair(Task):
    @staticmethod
    def get_closest(ui_index, ui_element, screen_elements):
        output_dict = dict()

        if len(screen_elements) <= TOTAL_SELECTED:
            output_dict = screen_elements
        else:
            distances_dict = dict()
            for ui_index_screen, ui_element_screen in screen_elements.items():
                x = ui_element_screen["x0"] - ui_element["x0"]
                y = ui_element_screen["y0"] - ui_element["y0"]
                total_distance = math.hypot(x, y)
                distances_dict[ui_index_screen] = total_distance

            while len(output_dict) < TOTAL_SELECTED:
                smallest = 100000
                smallest_id = 0

                for id_ui, distance in distances_dict.items():

                    if distance < smallest and id_ui not in output_dict:
                        smallest = distance
                        smallest_id = id_ui

                output_dict[smallest_id] = screen_elements[smallest_id]
            output_dict[ui_index] = ui_element

        return output_dict

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
                    "closest": self.get_closest(
                        ui_index, ui_element, screen_info["ui_elements"]
                    ),
                }
                total_pairs = total_pairs + 1
            index_query = index_query + 1

        logger.info(f"******** LARGEST UI TEXT: {largest_text} ********")
        logger.info(f"***** TOTAL UI ELEMENTS: {total_ui_elements}")
        logger.info(f"Total of pairs: {total_pairs}")
        logger.info(f"Total negative pairs: {total_negative_pairs}")
        logger.info(f"Total positive pairs: {total_positive_pairs}")

        return {"data": parsed_data, "mapping": mapping_query}
