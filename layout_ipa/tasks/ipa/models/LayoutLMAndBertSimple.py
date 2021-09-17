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


class LayoutLMAndBertSimpleConfig(PretrainedConfig):
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


class LayoutLMAndBertSimple(PreTrainedModel):
    config_class = LayoutLMAndBertSimpleConfig
    base_model_prefix = "layout_lm_bert"

    def __init__(self, config, screen_agg, combine_output, dropout, *args, **kwargs):
        super().__init__(config)

        self.screen_agg = screen_agg
        self.combine_output = combine_output
        self.model_ui1 = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )
        self.model_ui2 = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.dropout4 = nn.Dropout(p=dropout)

        self.linear_layer_instruction = nn.Linear(768, 1)
        self.linear_screen_fc = nn.Linear(768 * 5, 768)
        self.linear_screen = nn.Linear(256 * 5, 768)
        self.linear_ui_element = nn.Linear(768, 768)
        self.linear_combine = nn.Linear(768 * 4, 128)
        self.linear_combine_simple = nn.Linear(768, 128)
        self.linear_combine_double = nn.Linear(768 * 2, 128)
        self.linear_layer_ui = nn.Linear(768 * 5, 768)
        self.linear_layer_output = nn.Linear(128, 1)
        self.activation_ui1 = nn.Tanh()
        self.activation_ui2 = nn.Tanh()
        self.activation_instruction = nn.Tanh()

        self.deep_set = DeepSet(768, 5, 256)
        # self.linear_layer1 = nn.Linear(768 * 4, 1)
        # self.linear_layer2 = nn.Linear(512, 1)

    def forward(self, input_close_elements, input_ui):
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

        def get_screen_representations_deepset(input_close_elements):
            output_close_elements = self.model_ui2(**input_close_elements)[1]

            output_close_elements = output_close_elements.view(-1, 5, 768)

            output_close_elements = self.deep_set(output_close_elements)
            output_close_elements = self.dropout4(output_close_elements)

            output_close_elements = output_close_elements.view(-1, 5 * 256)

            screen_embedding = self.linear_screen(output_close_elements)

            output1 = self.dropout1(screen_embedding)

            return output1

        def get_screen_representations_fc(input_close_elements):
            output_close_elements = self.model_ui2(**input_close_elements)[1]

            output_close_elements = output_close_elements.view(-1, 5 * 768)

            screen_embedding = self.linear_screen_fc(output_close_elements)

            output1 = self.dropout1(screen_embedding)

            return output1

        def get_screen_representations_average(input_close_elements):
            output_close_elements = self.model_ui2(**input_close_elements)[1]

            output_close_elements = output_close_elements.view(-1, 5, 768)
            output_close_elements = output_close_elements.mean(1)

            screen_embedding = self.linear_screen_fc(output_close_elements)

            output1 = self.dropout1(screen_embedding)

            return output1

        def get_screen_representations_sum(input_close_elements):
            output_close_elements = self.model_ui2(**input_close_elements)[1]

            output_close_elements = output_close_elements.view(-1, 5, 768)
            output_close_elements = output_close_elements.sum(1)

            screen_embedding = self.linear_screen_fc(output_close_elements)

            output1 = self.dropout1(screen_embedding)

            return output1

        def get_ui_element_representations(input_ui):
            output_ui_model = self.model_ui1(**input_ui)
            ui_embedding = output_ui_model[1]
            ui_embedding = self.linear_ui_element(ui_embedding)
            output2 = self.dropout2(ui_embedding)

            return output2

        input_close_elements = convert_screen_elements_input_dimensions(
            input_close_elements
        )

        # help="0 - Deepset + FC; 1- FC; 2- Average; 3- Sum",
        if self.screen_agg == 0:
            output1 = get_screen_representations_deepset(input_close_elements)
        elif self.screen_agg == 1:
            output1 = get_screen_representations_fc(input_close_elements)
        elif self.screen_agg == 2:
            output1 = get_screen_representations_average(input_close_elements)
        elif self.screen_agg == 3:
            output1 = get_screen_representations_sum(input_close_elements)
        else:
            output1 = get_screen_representations_deepset(input_close_elements)

        output2 = get_ui_element_representations(input_ui)

        # help="0 - Matching; 1 - Concat; 2- Sum; 3- Mult",

        if self.combine_output == 0:
            output_combined = torch.cat(
                [output1, output2, torch.abs(output1 - output2), output1 * output2],
                dim=1,
            )
            output_combined = self.linear_combine(output_combined)
        elif self.combine_output == 1:
            output_combined = torch.cat([output1, output2], dim=1)
            output_combined = self.linear_combine_double(output_combined)
        elif self.combine_output == 2:
            output_combined = output1 + output2
            output_combined = self.linear_combine_simple(output_combined)
        elif self.combine_output == 3:
            output_combined = output1 * output2
            output_combined = self.linear_combine_simple(output_combined)
        else:
            output_combined = torch.cat(
                [output1, output2, torch.abs(output1 - output2), output1 * output2],
                dim=1,
            )
            output_combined = self.linear_combine(output_combined)

        output_combined = self.dropout3(output_combined)

        output = self.linear_layer_output(output_combined)

        return output


# class LayoutLMAndBertSimple(PreTrainedModel):
#     config_class = LayoutLMAndBertSimpleConfig
#     base_model_prefix = "layout_lm_bert"

#     def __init__(self, config, screen_agg, combine_output, dropout, *args, **kwargs):
#         super().__init__(config)

#         self.screen_agg = screen_agg
#         self.combine_output = combine_output
#         self.model_ui = AutoModel.from_pretrained(
#             LAYOUT_LM_MODEL, config=config.layout_lm
#         )

#         self.dropout1 = nn.Dropout(p=dropout)
#         self.dropout2 = nn.Dropout(p=dropout)
#         self.dropout3 = nn.Dropout(p=dropout)
#         self.dropout4 = nn.Dropout(p=dropout)

#         self.linear_layer_instruction = nn.Linear(768, 1)
#         self.linear_screen_fc = nn.Linear(768 * 5, 768)
#         self.linear_screen = nn.Linear(256 * 5, 768)
#         self.linear_ui_element = nn.Linear(768, 768)
#         self.linear_combine = nn.Linear(768 * 4, 128)
#         self.linear_combine_simple = nn.Linear(768, 128)
#         self.linear_combine_double = nn.Linear(768 * 2, 128)
#         self.linear_layer_ui = nn.Linear(768 * 5, 768)
#         self.linear_layer_output = nn.Linear(128, 1)
#         self.activation_ui1 = nn.Tanh()
#         self.activation_ui2 = nn.Tanh()
#         self.activation_instruction = nn.Tanh()

#         self.deep_set = DeepSet(768, 5, 256)
#         # self.linear_layer1 = nn.Linear(768 * 4, 1)
#         # self.linear_layer2 = nn.Linear(512, 1)

#     def forward(self, input_close_elements, input_ui):
#         output_ui_model = self.model_ui(**input_ui)
#         output1 = output_ui_model[1]

#         output1 = self.dropout1(output1)

#         output2 = self.model_ui(**input_close_elements)[1]

#         output2 = self.dropout2(output2)
#         output_combined = torch.cat(
#             [output1, output2, torch.abs(output1 - output2), output1 * output2], dim=1,
#         )
#         output_combined = self.linear_combine(output_combined)

#         output_combined = self.dropout3(output_combined)

#         output = self.linear_layer_output(output_combined)

#         return output


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=1024):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output),
        )

    def forward(self, X):
        X = self.enc(X)
        X = X.mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X
