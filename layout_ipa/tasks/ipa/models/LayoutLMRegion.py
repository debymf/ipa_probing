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


class LayoutLMRegionConfig(PretrainedConfig):
    model_type = "layout_lm_and_bert"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # assert (
        #     "layout_lm" in kwargs and "bert" in kwargs
        # ), "Layout Lm and Bert required."
        layout_lm_config = kwargs.pop("layout_lm")
        layout_lm_config_model_type = layout_lm_config.pop("model_type")

        # bert_config = kwargs.pop("bert")
        # bert_config_model_type = bert_config.pop("model_type")

        from transformers import AutoConfig

        self.layout_lm = AutoConfig.for_model(
            layout_lm_config_model_type, **layout_lm_config
        )
        # self.bert = AutoConfig.for_model(bert_config_model_type, **bert_config)
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


class LayoutLMRegion(PreTrainedModel):
    config_class = LayoutLMRegionConfig
    base_model_prefix = "layout_lm_bert"

    def __init__(self, config, screen_agg, combine_output, dropout, *args, **kwargs):
        super().__init__(config)

        self.screen_agg = screen_agg
        self.combine_output = combine_output
        self.model_ui = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )

        self.linear_screen_fc = nn.Linear(768, 1)

        # self.linear_layer1 = nn.Linear(768 * 4, 1)
        # self.linear_layer2 = nn.Linear(512, 1)

    def forward(self, input_elements):
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

            output = output_elements.view(-1, 25, 768)

            return output

        input_elements = convert_screen_elements_input_dimensions(input_elements)

        region_representations = get_screen_representations(input_elements)

        classification = self.linear_screen_fc(region_representations)

        classification = classification.squeeze(2)

        # # help="0 - Deepset + FC; 1- FC; 2- Average; 3- Sum",
        # if self.screen_agg == 0:
        #     output1 = get_screen_representations_deepset(input_close_elements)
        # elif self.screen_agg == 1:
        #     output1 = get_screen_representations_fc(input_close_elements)
        # elif self.screen_agg == 2:
        #     output1 = get_screen_representations_average(input_close_elements)
        # elif self.screen_agg == 3:
        #     output1 = get_screen_representations_sum(input_close_elements)
        # else:
        #     output1 = get_screen_representations_deepset(input_close_elements)

        # output2 = get_ui_element_representations(input_ui)

        # # help="0 - Matching; 1 - Concat; 2- Sum; 3- Mult",

        # if self.combine_output == 0:
        #     output_combined = torch.cat(
        #         [output1, output2, torch.abs(output1 - output2), output1 * output2],
        #         dim=1,
        #     )
        #     output_combined = self.linear_combine(output_combined)
        # elif self.combine_output == 1:
        #     output_combined = torch.cat([output1, output2], dim=1)
        #     output_combined = self.linear_combine_double(output_combined)
        # elif self.combine_output == 2:
        #     output_combined = output1 + output2
        #     output_combined = self.linear_combine_simple(output_combined)
        # elif self.combine_output == 3:
        #     output_combined = output1 * output2
        #     output_combined = self.linear_combine_simple(output_combined)
        # else:
        #     output_combined = torch.cat(
        #         [output1, output2, torch.abs(output1 - output2), output1 * output2],
        #         dim=1,
        #     )
        #     output_combined = self.linear_combine(output_combined)

        # output_combined = self.dropout3(output_combined)

        # output = self.linear_layer_output(output_combined)

        return classification

