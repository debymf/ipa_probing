from prefect import Task
from loguru import logger
from dynaconf import settings
import json
import random


class PrepareRicoScaEmbedding(Task):
    def run(self, file_location, type_instructions=[0, 1, 2, 3], num_negatives=20):
        """Parses the RicoSCA dataset for the embedding task.

        Args:
            file_location (str): location of the RicoSCA dataset
            type_instructions (List): Type of instructions that should be considered.
            where: 0 and 3 - Lexical Matching
            1 - Spatial (Relative to screen)
            2 - Spatial (Relative to other elements)

        Returns:
            Dict: preprocessed dict in the following format:
            
            {instruction: NL intruction,
            ui: DICT:
                    text: text of the ui element,
                    x0: bounding box x0,
                    x1: bouding box x1,
                    y0: bounding box y0,
                    y1: bounding box y1,
            label: 1 if UI element is refered in the instruction, 0 otherwise
            }
        """

        parsed_data = dict()
        logger.info("Preprocessing Rico SCA dataset")
        with open(file_location, "r") as f:
            input_data = json.load(f)

        number_of_screens = len(input_data)
        total_pairs = 0
        total_negative_pairs = 0
        total_positive_pairs = 0
        for _, screen_info in input_data.items():
            ui_elements_dict = dict()
            index_ui_element = 0

            for ui_element in screen_info["ui_obj_str_seq"]:
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

                if (
                    screen_info["instruction_rule_id"][index_instruction]
                    in type_instructions
                ):
                    parsed_data[total_pairs] = {
                        "instruction": instruction,
                        "ui": ui_elements_dict[selected_ui_element],
                        "label": 1,
                    }
                    total_pairs = total_pairs + 1
                    total_positive_pairs = total_positive_pairs + 1

                    if (len(ui_elements_dict) - 1) <= num_negatives:
                        for ui_index, ui_element in ui_elements_dict.items():
                            if ui_index != selected_ui_element:
                                parsed_data[total_pairs] = {
                                    "instruction": instruction,
                                    "ui": ui_element,
                                    "label": 0,
                                }
                                total_negative_pairs = total_negative_pairs + 1
                                total_pairs = total_pairs + 1
                    else:
                        ui_elements_list = list(ui_elements_dict)
                        ui_elements_list.remove(selected_ui_element)
                        keys = random.sample(ui_elements_list, num_negatives,)
                        negative_ui_elements = [ui_elements_dict[k] for k in keys]
                        for n in negative_ui_elements:
                            parsed_data[total_pairs] = {
                                "instruction": instruction,
                                "ui": n,
                                "label": 0,
                            }
                            total_pairs = total_pairs + 1
                            total_negative_pairs = total_negative_pairs + 1

                    index_instruction = index_instruction + 1

        logger.info(f"Number of different screens: {number_of_screens}.")
        logger.info(f"Total of pairs: {total_pairs}")
        logger.info(f"Total negative pairs: {total_negative_pairs}")
        logger.info(f"Total positive pairs: {total_positive_pairs}")

        return parsed_data
