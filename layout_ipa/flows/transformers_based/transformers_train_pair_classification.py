import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaPair
from layout_ipa.tasks.transformers_based.data_prep import PrepareTransformersPairTask
from layout_ipa.tasks.transformers_based.model_pipeline import TransformerPair
from sklearn.metrics import f1_score
from layout_ipa.util.evaluation import pair_evaluation_2d
from layout_ipa.tasks.datasets_parse.pixel_help import PreparePixelHelpPair
import argparse

parser = argparse.ArgumentParser()



train_path = settings["rico_sca"]["train"]
dev_path = settings["rico_sca"]["dev"]
test_path = settings["rico_sca"]["test"]

## Uncomment this if you want to test for pixel_help
#test_path = settings["pixel_help"]


parser.add_argument(
    "--model",
    metavar="model",
    type=str,
    nargs="?",
    help="LLMs used for probing",
    default="bert-base-uncased",
)


cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"./cache/datasets/rico/"),
)

args = parser.parse_args()

# Change the instruction type that you require here
INSTRUCTION_TYPE = [0,1,2,3]
#  where: 0 and 3 - Extractive
#             1 - Absolute
#             2 - Relative
prepare_rico_task = PrepareRicoScaPair()
prepare_pixel_help_task = PreparePixelHelpPair()
prepare_rico_transformer_task = PrepareTransformersPairTask()
transformer_trainer_task = TransformerPair()


with Flow("Running the Transformers for Pair Classification") as flow1:
    with tags("train"):
        train_input = prepare_rico_task(train_path, type_instructions=INSTRUCTION_TYPE)
        train_dataset = prepare_rico_transformer_task(train_input["data"], model=args.model)
    with tags("dev"):
        dev_input = prepare_rico_task(dev_path, type_instructions=INSTRUCTION_TYPE)
        dev_dataset = prepare_rico_transformer_task(dev_input["data"],model=args.model)
    with tags("test"):
        test_input = prepare_rico_task(test_path, type_instructions=INSTRUCTION_TYPE)
        ## Uncomment this if you want to test for pixel help
        # test_input = prepare_pixel_help_task(test_path)
        test_dataset = prepare_rico_transformer_task(test_input["data"], model=args.model)
    transformer_trainer_task(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        mapping_dev=dev_input["mapping"],
        mapping_test=test_input["mapping"],
        bert_model = args.model,
        task_name="transformer_pair_rico",
        output_dir="./cache/transformer_pair_rico/",
        mode="train", # For test only, change this to test
        eval_fn=pair_evaluation_2d,
    )


FlowRunner(flow=flow1).run()
