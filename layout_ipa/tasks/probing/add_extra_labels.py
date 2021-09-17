from prefect import Task
from loguru import logger
from dynaconf import settings
import json
import random
import math
from transformers import AutoTokenizer

TOTAL_SELECTED = 15

LARGEST_Y = 1000
LARGEST_X = 1000


class AddExtraLabelsTask(Task):
    def run(self, dataset):
        """Parses the RicoSCA dataset for the pair classification task.

        Args:
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

        """

        correct_element = dict()
        for id_i, content in dataset.items():
            if content["label"] == 1:
                correct_element[content["id_query"]] = {
                    "x0": content["ui"]["x0"],
                    "y0": content["ui"]["y0"],
                }

        new_dict = dict()
        for id_i, content in dataset.items():
            relative_x0 = correct_element[content["id_query"]]["x0"]
            relative_y0 = correct_element[content["id_query"]]["y0"]
            new_dict[id_i] = content
            y0 = content["ui"]["y0"]
            x0 = content["ui"]["x0"]
            new_dict[id_i]["is_top"] = 1 if y0 > (LARGEST_Y / 2) else 0
            new_dict[id_i]["is_right"] = 1 if x0 > (LARGEST_X / 2) else 0
            if ((x0 - relative_x0) == 0) and ((y0 - relative_y0) == 0):
                new_dict[id_i]["relative_is_top"] = -1
                new_dict[id_i]["relative_is_right"] = -1
            else:
                new_dict[id_i]["relative_is_top"] = 1 if y0 > relative_y0 else 0
                new_dict[id_i]["relative_is_right"] = 1 if x0 > relative_x0 else 0

        return new_dict

