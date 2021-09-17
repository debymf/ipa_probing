import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoConfig
from torch.autograd import Variable
from dynaconf import settings
from transformers import PreTrainedModel
import os
import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.file_utils import WEIGHTS_NAME

torch.utils.backcompat.broadcast_warning.enabled = True
torch.set_printoptions(threshold=5000)

BERT_MODEL = "bert-base-uncased"
LAYOUT_LM_MODEL = "microsoft/layoutlm-base-uncased"


logger = logging.get_logger(__name__)


class LayoutLMSelectModelConfig(PretrainedConfig):
    model_type = "layout_lm_and_bert"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # assert (
        #     "layout_lm" in kwargs and "bert" in kwargs
        # ), "Layout Lm and Bert required."
        layout_lm_config = kwargs.pop("layout_lm")
        layout_lm_config_model_type = layout_lm_config.pop("model_type")

        bert_config = kwargs.pop("bert")
        bert_config_model_type = bert_config.pop("model_type")

        from transformers import AutoConfig

        self.layout_lm = AutoConfig.for_model(
            layout_lm_config_model_type, **layout_lm_config
        )
        self.bert = AutoConfig.for_model(bert_config_model_type, **bert_config)
        # self.is_encoder_decoder = True

    @classmethod
    def from_layout_lm_bert_configs(
        cls, layout_lm_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:

        return cls(layout_lm=layout_lm_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["layout_lm"] = self.layout_lm.to_dict()

        output["model_type"] = self.__class__.model_type
        return output


class LayoutLMSelectModel(PreTrainedModel):
    config_class = LayoutLMSelectModelConfig
    base_model_prefix = "layout_lm_bert"

    def __init__(self, config, screen_agg, combine_output, dropout, *args, **kwargs):
        super().__init__(config)

        self.screen_agg = screen_agg
        self.combine_output = combine_output
        self.model_ui = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )
        for param in self.model_ui.parameters():
            param.requires_grad = False

        self.linear_screen_fc = nn.Linear(768, 1)

        # self.linear_layer1 = nn.Linear(768 * 4, 1)
        # self.linear_layer2 = nn.Linear(512, 1)

    def forward(self, input_instruction, input_screen):
        def convert_screen_elements_input_dimensions(input_close_elements):

            input_close_elements["input_ids"] = input_close_elements["input_ids"].view(
                -1, input_close_elements["input_ids"].size(-1)
            )

            input_close_elements["attention_mask"] = input_close_elements[
                "attention_mask"
            ].view(-1, input_close_elements["attention_mask"].size(-1))

            input_close_elements["token_type_ids"] = input_close_elements[
                "token_type_ids"
            ].view(-1, input_close_elements["token_type_ids"].size(-1))

            input_close_elements["bbox"] = input_close_elements["bbox"].view(
                -1, input_close_elements["bbox"].size(-2), 4
            )

            return input_close_elements

        def get_screen_representations(input_elements):
            output_elements = self.model_ui(**input_elements)[1]

            output = output_elements.view(-1, 300, 768)

            return output

        # input_screen = convert_screen_elements_input_dimensions(input_screen)

        # screen_representation = get_screen_representations(input_screen)

        classification = self.linear_screen_fc(input_screen)

        classification = classification.squeeze(2)

        return classification

