import os
import random
import torch.nn as nn
import numpy as np
import torch
from loguru import logger
from overrides import overrides
from prefect import Task
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from scipy.special import expit, logit
from torch.utils.data import WeightedRandomSampler
import json
from sklearn.metrics import precision_recall_fscore_support, f1_score
from layout_ipa.tasks.ipa.models import (
    LayoutLMAndBert,
    LayoutLMAndBertConfig,
)
from sklearn.metrics import f1_score
import pandas as pd
import csv

np.set_printoptions(threshold=np.inf)


class GetVectorsLayoutLMProbing(Task):
    @overrides
    def run(self, dataset, model_location):
        torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_gpu = 1
        device = "cpu"

        self.logger.info(f"GPUs used {n_gpu}")

        logger.info("Obtaining output vectors.")

        model_config = AutoConfig.from_pretrained(model_location)
        # layout_lm_config = AutoConfig.from_pretrained(f"{output_dir}/{task_name}")
        logger.info(f"Loading from {model_location}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_location, config=model_config
        )

        test_data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        model.to(device)
        output_dict = self.eval(model, test_data_loader, dataset, device, n_gpu,)

        return output_dict

    def eval(
        self, model, dataloader, dataset, device, n_gpu,
    ):

        nb_eval_steps = 0
        out_embedding = None
        out_label_ids = None
        instruction_type = list()
        instruction_text = list()
        output_dict = dict()
        ui_text = list()
        output_dict["representation"] = list()
        output_dict["labels"] = list()
        output_dict["instruction"] = list()
        output_dict["ui_text"] = list()
        output_dict["type"] = list()
        output_dict["is_top"] = list()
        output_dict["is_right"] = list()
        output_dict["relative_is_top"] = list()
        output_dict["relative_is_right"] = list()
        for batch in tqdm(dataloader, desc="Evaluating"):

            model.eval()

            with torch.no_grad():

                inputs_layoutlm = {
                    "input_ids": batch[0].to(device),
                    "attention_mask": batch[1].to(device),
                    "token_type_ids": batch[2].to(device),
                    "bbox": batch[3].to(device),
                }

                labels = batch[4]

                instructions_type = batch[6].tolist()
                instruction_text = batch[7]
                ui_text = batch[8]
                is_top = batch[9].tolist()
                is_right = batch[10].tolist()
                relative_is_top = batch[11].tolist()
                relative_is_right = batch[12].tolist()

                last_hidden_states = model.layoutlm(**inputs_layoutlm)[0]
                cls_hidden_state = last_hidden_states[:, 0, :]

            nb_eval_steps += 1
            if out_embedding is None:
                out_embedding = cls_hidden_state.detach().cpu().numpy()

                out_label_ids = labels.detach().cpu().numpy()

            else:
                out_embedding = np.append(
                    out_embedding, cls_hidden_state.detach().cpu().numpy(), axis=0
                )

                out_label_ids = np.append(out_label_ids, labels.cpu().numpy(), axis=0)
            output_dict["ui_text"].extend(ui_text)
            output_dict["instruction"].extend(instruction_text)
            output_dict["type"].extend(instructions_type)
            output_dict["is_top"].extend(is_top)
            output_dict["is_right"].extend(is_right)
            output_dict["relative_is_top"].extend(is_top)
            output_dict["relative_is_right"].extend(is_right)
        output_dict["labels"].extend(out_label_ids.tolist())
        output_dict["representation"].extend(out_embedding.tolist())
        # eval_loss = eval_loss / nb_eval_steps

        return output_dict
