from prefect import Task
from loguru import logger
from dynaconf import settings
import json
import random
import numpy as np
from scipy import stats
import pandas as pd
import math
from tqdm import tqdm

NUM_REGIONS = 9
# REGION_MATRIX = [
#     [0, 1, 2, 3, 4],
#     [5, 6, 7, 8, 9],
#     [10, 11, 12, 13, 14],
#     [15, 16, 17, 18, 19],
#     [20, 21, 22, 23, 24],
# ]
# REGION_BINS = np.array([0.0, 200.0, 400.0, 600.0, 800.0, 1001.0])
REGION_MATRIX = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
REGION_BINS = np.array([0.0, 333.0, 666.0, 1001.0])


class PrepareRicoScaRegion(Task):
    def run(self, input_data):
        """Parses the RicoSCA dataset for the UI selection task.

        Args:
            Dict: preprocessed dict in the following format:
            
            {instruction: NL intruction,
            ui: DICT:
                    text: text of the ui element,
                    x0: bounding box x0,
                    x1: bouding box x1,
                    y0: bounding box y0,
                    y1: bounding box y1,
            label: From the list of UI elements, obtain the one reffered in the sentence.
            }

        Returns:
        {instruction: NL intruction,
            regions
            label_regions
            ui: DICT:
                    text: text of the ui element,
                    x0: bounding box x0,
                    x1: bouding box x1,
                    y0: bounding box y0,
                    y1: bounding box y1,
            label: From the list of UI elements, obtain the one reffered in the sentence.
            }

            
        """

        parsed_data = dict()
        largest = 0
        logger.info("Separating elements by regions")

        for input_id, input_content in tqdm(input_data.items()):

            parsed_data[input_id] = dict()
            parsed_data[input_id]["instruction"] = input_content["instruction"]
            parsed_data[input_id]["regions"] = dict()
            for i in range(0, NUM_REGIONS):
                parsed_data[input_id]["regions"][i] = dict()

            # if len(input_content["ui"]) > NUM_REGIONS:
            #     try:
            #         x_list = list()
            #         y_list = list()
            #         for ui_id, ui_content in input_content["ui"].items():
            #             x_list.append(ui_content["x0"])
            #             y_list.append(ui_content["y0"])

            #         qc_x = pd.qcut(
            #             x_list,
            #             q=int(math.sqrt(NUM_REGIONS)),
            #             precision=1,
            #             duplicates="drop",
            #         )
            #         qc_y = pd.qcut(
            #             y_list,
            #             q=int(math.sqrt(NUM_REGIONS)),
            #             precision=1,
            #             duplicates="drop",
            #         )

            #         x_bins = qc_x.codes
            #         y_bins = qc_y.codes

            #         i = 0
            #         for ui_id, ui_content in input_content["ui"].items():
            #             parsed_data[input_id]["regions"][
            #                 REGION_MATRIX[x_bins[i]][y_bins[i]]
            #             ][ui_id] = ui_content
            #             i += 1
            #     except IndexError:
            #         for ui_id, ui_content in input_content["ui"].items():
            #             bin_x = np.digitize(ui_content["x0"], REGION_BINS) - 1
            #             bin_y = np.digitize(ui_content["y0"], REGION_BINS) - 1
            #             parsed_data[input_id]["regions"][REGION_MATRIX[bin_x][bin_y]][
            #                 ui_id
            #             ] = ui_content

            # else:
            for ui_id, ui_content in input_content["ui"].items():
                bin_x = np.digitize(ui_content["x0"], REGION_BINS) - 1
                bin_y = np.digitize(ui_content["y0"], REGION_BINS) - 1
                parsed_data[input_id]["regions"][REGION_MATRIX[bin_x][bin_y]][
                    ui_id
                ] = ui_content

            label_region = -1
            for i in range(0, NUM_REGIONS):
                if input_content["label"] in parsed_data[input_id]["regions"][i]:
                    label_region = i
                if len(parsed_data[input_id]["regions"][i]) > largest:
                    largest = len(parsed_data[input_id]["regions"][i])

            parsed_data[input_id]["ui"] = input_content["ui"]
            parsed_data[input_id]["label_region"] = label_region
            parsed_data[input_id]["label"] = input_content["label"]

        logger.info(f"LARGEST: {largest}")
        return parsed_data
