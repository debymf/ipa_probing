from prefect import Task
from loguru import logger
from dynaconf import settings
import json
import random
import math
from transformers import AutoTokenizer

TOTAL_SELECTED = 15


class PrepareRicoScaScreenPair(Task):
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

    def run(self, file_location, type_instructions=[0, 1, 2, 3], limit=None):
        """Parses the RicoSCA dataset for the pair classification task.

        Args:
            file_location (str): location of the RicoSCA dataset
            type_instructions (List): Type of instructions that should be considered.
            where: 0 and 3 - Lexical Matching
            1 - Spatial (Relative to screen)
            2 - Spatial (Relative to other elements)

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

        parsed_data = dict()
        logger.info("Preprocessing Rico SCA dataset")
        with open(file_location, "r") as f:
            input_data = json.load(f)

        number_of_screens = len(input_data)
        total_instructions = 0
        total_pairs = 0
        total_negative_pairs = 0
        total_positive_pairs = 0
        index_query = 0
        total_ui_elements = 0
        largest_text = 0

        tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        mapping_query = dict()
        for _, screen_info in input_data.items():
            ui_elements_dict = dict()
            index_ui_element = 0

            for ui_element in screen_info["ui_obj_str_seq"]:
                size_ui = len(tokenizer.tokenize(ui_element))
                if size_ui > 128:
                    largest_text = largest_text + 1
                total_ui_elements = total_ui_elements + 1
                ui_elements_dict[index_ui_element] = {
                    "text": ui_element,
                    "x0": screen_info["ui_obj_cord_x_seq"][index_ui_element * 2] * 1000,
                    "x1": screen_info["ui_obj_cord_x_seq"][(2 * index_ui_element) + 1]
                    * 1000,
                    "y0": screen_info["ui_obj_cord_y_seq"][index_ui_element * 2] * 1000,
                    "y1": screen_info["ui_obj_cord_y_seq"][(2 * index_ui_element) + 1]
                    * 1000,
                }

                index_ui_element = index_ui_element + 1

            index_instruction = 0

            for instruction in screen_info["instruction_str"]:

                selected_ui_element = screen_info["ui_target_id_seq"][index_instruction]
                mapping_query[index_query] = selected_ui_element
                if (
                    screen_info["instruction_rule_id"][index_instruction]
                    in type_instructions
                ):
                    for ui_index, ui_element in ui_elements_dict.items():

                        if ui_index == selected_ui_element:
                            label_ui = 1
                            total_positive_pairs = total_positive_pairs + 1
                        else:
                            total_negative_pairs = total_negative_pairs + 1
                            label_ui = 0

                        if "\n" in instruction:
                            instruction = instruction.replace("\n", "")
                        parsed_data[total_pairs] = {
                            "id_query": index_query,
                            "ui_position": ui_index,
                            "instruction": instruction,
                            "ui": ui_element,
                            "label": label_ui,
                            "closest": self.get_closest(
                                ui_index, ui_element, ui_elements_dict
                            ),
                        }
                        total_pairs = total_pairs + 1

                    index_query = index_query + 1
                    index_instruction = index_instruction + 1
                    total_instructions += 1

                if limit and total_instructions >= limit:
                    break

            if limit and total_instructions >= limit:
                break

        logger.info(f"******** LARGEST UI TEXT: {largest_text} ********")
        logger.info(f"***** TOTAL UI ELEMENTS: {total_ui_elements}")
        logger.info(f"Number of different screens: {number_of_screens}.")
        logger.info(f"Total of pairs: {total_pairs}")
        logger.info(f"Total negative pairs: {total_negative_pairs}")
        logger.info(f"Total positive pairs: {total_positive_pairs}")

        return {"data": parsed_data, "mapping": mapping_query}
