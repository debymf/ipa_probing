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

np.set_printoptions(threshold=np.inf)

BERT_MODEL = "bert-base-uncased"
LAYOUT_LM_MODEL = "microsoft/layoutlm-base-uncased"


class BertandLayoutLMTrainer(Task):
    def __init__(self, **kwargs):
        super(BertandLayoutLMTrainer, self).__init__(**kwargs)
        self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 16)
        self.cuda = kwargs.get("cuda", True)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.num_train_epochs = kwargs.get("num_train_epochs", 20)
        self.learning_rate = kwargs.get("learning_rate", 5e-5)
        self.weight_decay = kwargs.get("weight_decay", 0.1)
        self.adam_epsilon = kwargs.get("adam_epsilon", 1e-8)
        self.warmup_steps = kwargs.get("warmup_steps", 0)
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.logging_steps = kwargs.get("logging_steps", 5)
        self.args = kwargs

    def set_seed(self, n_gpu, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    @overrides
    def run(
        self,
        train_dataset,
        dev_dataset,
        test_dataset,
        mapping_dev,
        mapping_test,
        task_name,
        output_dir,
        bert_model="bert-base-uncased",
        num_labels=2,
        mode="train",
        eval_fn=None,
        save_optimizer=False,
        eval_params={},
    ):
        torch.cuda.empty_cache()
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        n_gpu = torch.cuda.device_count()

        self.logger.info(f"GPUs used {n_gpu}")

        train_batch_size = self.per_gpu_batch_size * max(1, n_gpu)

        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True,
        )
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=train_batch_size, shuffle=False,
        )

        self.set_seed(n_gpu)

        criterion = nn.BCEWithLogitsLoss()

        outputs = {}
        if mode == "train":
            logger.info("Running train mode")
            bert_config = AutoConfig.from_pretrained(BERT_MODEL)
            layout_lm_config = AutoConfig.from_pretrained(LAYOUT_LM_MODEL)

            config_dict = {
                "layout_lm_config": layout_lm_config,
                "bert_config": bert_config,
            }

            config = LayoutLMAndBertConfig.from_layout_lm_bert_configs(**config_dict)

            model = LayoutLMAndBert(config=config)
            model = model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            epoch_results = self.train(
                model,
                train_dataloader,
                dev_dataloader,
                dev_dataset,
                mapping_dev,
                device,
                criterion,
                n_gpu,
                eval_fn,
                f"{output_dir}/{task_name}",
                save_optimizer,
                eval_params,
                bert_model=bert_model,
            )
            outputs["epoch_results "] = epoch_results
        logger.info("Running evaluation mode")

        model_config = LayoutLMAndBertConfig.from_pretrained(
            f"{output_dir}/{task_name}"
        )
        # layout_lm_config = AutoConfig.from_pretrained(f"{output_dir}/{task_name}")
        logger.info(f"Loading from {output_dir}/{task_name}")

        model = LayoutLMAndBert.from_pretrained(
            f"{output_dir}/{task_name}", config=model_config
        )

        model.to(device)
        score = self.eval(
            criterion,
            model,
            dev_dataloader,
            dev_dataset,
            mapping_dev,
            device,
            n_gpu,
            eval_fn,
            eval_params,
            mode="dev",
            bert_model=bert_model,
        )
        outputs["dev"] = {
            "score": score,
        }
        if test_dataset is not None:
            test_data_loader = DataLoader(
                test_dataset, batch_size=train_batch_size, shuffle=False
            )
            score = self.eval(
                criterion,
                model,
                test_data_loader,
                test_dataset,
                mapping_test,
                device,
                n_gpu,
                eval_fn,
                eval_params,
                mode="test",
                bert_model=bert_model,
            )
            outputs["test"] = {
                "score": score,
            }

        return outputs

    def train(
        self,
        model,
        train_dataloader,
        dev_dataloader,
        dev_dataset,
        mapping_dev,
        device,
        criterion,
        n_gpu,
        eval_fn,
        output_dir,
        save_optimizer,
        eval_params,
        bert_model,
    ):
        results = {}
        best_score = 0.0
        t_total = (
            len(train_dataloader)
            // self.gradient_accumulation_steps
            * self.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
        ]

        optimizer = AdamW(
            model.parameters(), lr=self.learning_rate, eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total,
        )

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(self.num_train_epochs), desc="Epoch",
        )

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            epoch_loss = 0.0
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()

                batch = tuple(t.to(device) for t in batch)
                inputs_bert = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                inputs_layout_lm = {
                    "input_ids": batch[3],
                    "attention_mask": batch[4],
                    "token_type_ids": batch[5],
                    "bbox": batch[6],
                }

                outputs, _ = model(inputs_bert, inputs_layout_lm)

                labels = batch[7]
                labels = labels.type_as(outputs)

                # preds = outputs.detach().cpu().numpy()
                # preds = np.argmax(preds, axis=1)

                # print("\n\n")
                # print("=====================================")
                # print("*** PREDS ****")
                # print(preds)
                # print("\n\n")

                # print("**** LABEL *****")
                # print(labels.detach().cpu().numpy())
                # print("\n\n")

                # print("**** SCORE ******")
                # score = eval_fn(preds, labels.detach().cpu().numpy())
                # print(score)
                # print("\n\n")
                # print("\n\n")

                loss = criterion(outputs, labels.unsqueeze(1))

                if n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                epoch_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        loss_scalar = (tr_loss - logging_loss) / self.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        epoch_iterator.set_description(
                            f"Loss :{loss_scalar} LR: {learning_rate_scalar}"
                        )
                        logging_loss = tr_loss

            logger.debug(f"TRAINING LOSS: {epoch_loss}")
            score = self.eval(
                criterion,
                model,
                dev_dataloader,
                dev_dataset,
                mapping_dev,
                device,
                n_gpu,
                eval_fn,
                eval_params,
                mode="dev",
                bert_model=bert_model,
            )
            results[epoch] = score
            with torch.no_grad():
                if score >= best_score:
                    logger.info(f"Storing the new model with score: {score}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)

                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    logger.info(f"Saving model checkpoint to {output_dir}")
                    if save_optimizer:
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir
                        )
                    best_score = score

        return results

    def eval(
        self,
        criterion,
        model,
        dataloader,
        dataset,
        mapping,
        device,
        n_gpu,
        eval_fn,
        eval_params,
        mode,
        bert_model="bert",
    ):
        if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                query_ids = batch[8]
                ui_positions = batch[9]

                inputs_bert = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                inputs_layout_lm = {
                    "input_ids": batch[3],
                    "attention_mask": batch[4],
                    "token_type_ids": batch[5],
                    "bbox": batch[6],
                }

                outputs_logits, outputs = model(inputs_bert, inputs_layout_lm)

                labels = batch[7]

                loss = criterion(
                    outputs_logits, labels.type_as(outputs_logits).unsqueeze(1)
                )

                eval_loss += loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                index_queries = query_ids.detach().cpu().numpy()
                preds = outputs.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_ui = ui_positions.detach().cpu().numpy()

            else:
                preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)

                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

                index_queries = np.append(
                    index_queries, query_ids.detach().cpu().numpy(), axis=0
                )

                all_ui = np.append(all_ui, ui_positions.detach().cpu().numpy(), axis=0)
        # eval_loss = eval_loss / nb_eval_steps
        logger.success(f"EVAL LOSS: {eval_loss}")
        score = None
        if eval_fn is not None:

            preds_parsed = preds.squeeze(1)

            # print("**** PREDS ****")
            # print(preds)
            # input()
            # print("**** Preds Parsed ****")
            # print(preds_parsed)
            # input()
            score = eval_fn(preds_parsed, index_queries, all_ui, mapping)

            logger.info(
                f"F1-Score (Sanity check): {f1_score(y_true=out_label_ids, y_pred=np.where(preds_parsed > 0.5, 1, 0))}"
            )

            # if mode == "test":
            #     out_preds = {"preds": preds.tolist(), "gold": out_label_ids.tolist()}
            #     with open(f"./cache/output/bin_preds.json", "w") as fp:
            #         json.dump(out_preds, fp)

            logger.info(f"Score:{score}")

        return score
