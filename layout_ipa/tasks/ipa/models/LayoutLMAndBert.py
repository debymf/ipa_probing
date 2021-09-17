import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoConfig
from torch.autograd import Variable
from .bidaf import BidafAttn
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


class LayoutLMAndBertConfig(PretrainedConfig):
    model_type = "layout_lm_and_bert"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "layout_lm" in kwargs and "bert" in kwargs
        ), "Layout Lm and Bert required."
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
        cls, layout_lm_config: PretrainedConfig, bert_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:

        # logger.info(
        #     "Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config"
        # )
        # decoder_config.is_decoder = True
        # decoder_config.add_cross_attention = True

        return cls(
            layout_lm=layout_lm_config.to_dict(), bert=bert_config.to_dict(), **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["layout_lm"] = self.layout_lm.to_dict()
        output["bert"] = self.bert.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class LayoutLMAndBert(PreTrainedModel):
    config_class = LayoutLMAndBertConfig
    base_model_prefix = "layout_lm_bert"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.model_instruction = AutoModel.from_pretrained(
            BERT_MODEL, config=config.bert
        )

        self.model_ui = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )

        self.bidaf = BidafAttn(768)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.dropout3 = nn.Dropout(p=0.3)
        self.instruction_mlp = MLP(768, 256)
        self.ui_mlp = MLP(768, 256)

        self.combination_mlp = MLP(768 * 2, 768)

        self.linear_layer_instruction = nn.Linear(768, 1)
        self.linear_layer_ui = nn.Linear(768 * 4, 1)
        self.linear_layer_output = nn.Linear(128 * 3072, 1)
        self.activation_ui1 = nn.Tanh()
        self.activation_ui2 = nn.Tanh()
        self.activation_instruction = nn.Tanh()

    def forward(self, input_instructions, input_ui):

        output1 = self.model_instruction(**input_instructions)[0]
        instruction_representation = self.dropout1(output1)

        # output2 = self.model_instruction(**input_ui_text)[1]
        # ui_text_representation = self.dropout2(output2)

        # both_representations = self.bidaf(output1, output2)

        # simple = self.linear_layer_ui(both_representations)

        # simple = simple.squeeze(2)

        # both_representations = torch.cat(
        #    (instruction_representation, ui_text_representation), dim=1
        # )

        # both_representations = torch.cat(
        #     [output1, output2, torch.abs(output1 - output2), output1 * output2], dim=1
        # )

        # output = self.linear_layer_output(both_representations)

        # instruction_representation = self.dropout1(instruction_representation)
        # instruction_mlp_output = self.instruction_mlp(instruction_representation)

        output_ui_model = self.model_ui(**input_ui)
        ui_element_representation = output_ui_model[0]
        ui_element_representation = self.dropout2(ui_element_representation)

        # ui_mlp_output = self.ui_mlp(ui_element_representation)

        c2q_attention, q2c_attention = self.bidaf(
            instruction_representation, ui_element_representation
        )

        combined_features = torch.cat(
            (
                ui_element_representation,
                c2q_attention,
                torch.mul(ui_element_representation, c2q_attention),
                torch.mul(ui_element_representation, q2c_attention),
            ),
            dim=-1,
        )  # (N,T,4d)

        both_representations = combined_features.view(-1, 128 * 3072)

        # both_mlp_output = self.combination_mlp(both_representations)
        output = self.linear_layer_output(both_representations)

        predictions = torch.sigmoid(output)

        return output, predictions


class MLP(nn.Module):
    """mlp can specify number of hidden layers and hidden layer channels"""

    def __init__(
        self,
        input_dim,
        output_dim,
        act="relu",
        num_hidden_lyr=2,
        dropout_prob=0.5,
        return_layer_outs=False,
        hidden_channels=None,
        bn=False,
    ):
        super().__init__()
        self.out_dim = output_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.return_layer_outs = return_layer_outs
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels"
            )
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.act_name = act
        self.activation = create_act(act)
        self.layers = nn.ModuleList(
            list(
                map(
                    self.weight_init,
                    [
                        nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                        for i in range(len(self.layer_channels) - 2)
                    ],
                )
            )
        )
        final_layer = nn.Linear(self.layer_channels[-2], self.layer_channels[-1])
        self.weight_init(final_layer, activation="linear")
        self.layers.append(final_layer)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList(
                [torch.nn.BatchNorm1d(dim) for dim in self.layer_channels[1:-1]]
            )

    def weight_init(self, m, activation=None):
        if activation is None:
            activation = self.act_name
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
        return m

    def forward(self, x):
        """
        :param x: the input features
        :return: tuple containing output of MLP,
                and list of inputs and outputs at every layer
        """
        layer_inputs = [x]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(self.dropout(output))

        # model.store_layer_output(self, layer_inputs[-1])
        if self.return_layer_outs:
            return layer_inputs[-1], layer_inputs
        else:
            return layer_inputs[-1]


def calc_mlp_dims(input_dim, division=2, output_dim=1):
    dim = input_dim
    dims = []
    while dim > output_dim:
        dim = dim // division
        dims.append(int(dim))
    dims = dims[:-1]
    return dims


def create_act(act, num_parameters=None):
    if act == "relu":
        return nn.ReLU()
    elif act == "prelu":
        return nn.PReLU(num_parameters)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":

        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError("Unknown activation function {}".format(act))


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def hf_loss_func(inputs, classifier, labels, num_labels, class_weights):
    logits = classifier(inputs)
    if type(logits) is tuple:
        logits, layer_outputs = logits[0], logits[1]
    else:  # simple classifier
        layer_outputs = [inputs, logits]
    if labels is not None:
        if num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss(weight=class_weights)
            labels = labels.long()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
    else:
        return None, logits, layer_outputs

    return loss, logits, layer_outputs
