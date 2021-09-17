import json
import math
from collections import defaultdict
from typing import Dict
import numpy as np
from loguru import logger
from prefect import Task
from tqdm import tqdm


def pair_evaluation_2d(preds, q_ids, ui_ids, mappings):
    preds = preds[:, 1]
    scores = dict()
    choices = dict()

    for i in range(0, len(preds)):
        if q_ids[i] not in scores:
            scores[q_ids[i]] = preds[i]
            choices[q_ids[i]] = ui_ids[i]
        else:
            if preds[i] > scores[q_ids[i]]:
                scores[q_ids[i]] = preds[i]
                choices[q_ids[i]] = ui_ids[i]

    total = 0
    total_correct = 0

    for id_q, choice in choices.items():
        correct_choice = mappings[id_q]
        if choice == correct_choice:
            total_correct = total_correct + 1
        total = total + 1

        acc = total_correct / total

    return acc


def pair_evaluation(preds, q_ids, ui_ids, mappings):
    # preds = preds[:, 1]
    scores = dict()
    choices = dict()

    for i in range(0, len(preds)):
        if q_ids[i] not in scores:
            scores[q_ids[i]] = preds[i]
            choices[q_ids[i]] = ui_ids[i]
        else:
            if preds[i] > scores[q_ids[i]]:
                scores[q_ids[i]] = preds[i]
                choices[q_ids[i]] = ui_ids[i]

    total = 0
    total_correct = 0

    for id_q, choice in choices.items():
        correct_choice = mappings[id_q]
        if choice == correct_choice:
            total_correct = total_correct + 1
        total = total + 1

        acc = total_correct / total

    return acc


def pair_evaluation_vector(preds, q_ids, ui_ids, mappings):
    scores = dict()
    choices = dict()

    for i in range(0, len(preds)):
        if q_ids[i] not in scores:
            scores[q_ids[i]] = preds[i]
            choices[q_ids[i]] = ui_ids[i]
        else:
            if preds[i] > scores[q_ids[i]]:
                scores[q_ids[i]] = preds[i]
                choices[q_ids[i]] = ui_ids[i]

    total = 0
    total_correct = 0

    for id_q, choice in choices.items():
        correct_choice = mappings[id_q]
        if choice == correct_choice:
            total_correct = total_correct + 1
        total = total + 1

        acc = total_correct / total

    return acc

