# Probing LLMs for UI element Grounding

Welcome! :) This is the repository for the paper **Grounding Natural Language Instructions: Can Large Language Models Capture Spatial Information?**. If you find the code useful, please cite our paper!

## Installing the requirements

```
pip install -r requirements.txt
```

## Running Model

The datasets with the splits used in the paper can be found inside the ```data``` folder.

Running LLMs for UI grounding:

```
python -m  layout_ipa.flows.transformers_based.transformers_train_pair_classification --model=[MODEL]
```

Replace **[MODEL]** with the model name. In this work we teste for ```bert-base-uncased``` and ```roberta-base```, but it will likely work for other models with minimal intervation


Running ```Layout-LM``` for UI grounding:

```
python -m  layout_ipa.flows.layout_lm.layout_lm_train_pair_classification
```

We used LayoutLMv2 for our experiments (```microsoft/layoutlmv2-base-uncased```).





