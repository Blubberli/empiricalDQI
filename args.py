# based on transformers/examples/pytorch/text-classification/run_glue.py

import os
import sys
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments
from transformers.file_utils import ExplicitEnum


class QualityDimension(ExplicitEnum):
    """The four possible labels that we want to predict"""
    JUSTIFICATION = "jlev"
    COMMON_GOOD = "jcon"
    RESPECT_GROUP = "resp_gr"
    INTERACTIVITY = "int1"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    quality_dim: QualityDimension = field(
        metadata={
            "help": "The quality dimension that should be predicted."
        }
    )

    task1: Optional[str] = field(
                                 metadata={"help": "the column which stores the quality dim of the first task"}, default="jlev")
    task2: Optional[str] = field(
                                 metadata={"help": "the column which stores the quality dim of the second task"}, default="jcon")
    task3: Optional[str] = field(
                                 metadata={"help": "the column which stores the quality dim of the second task"}, default="int1")
    task4: Optional[str] = field(
                                 metadata={"help": "the column which stores the quality dim of the second task"}, default="resp_gr")
    text_col: Optional[str] = field(default="cleaned_comment", metadata={"help": "the column which stores the text"})
    data_dir: Optional[str] = field(
        default=str('5foldStratified/jlev')
    )

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_feats: bool = field(
        default=False, metadata={"help": "Whether to use additional features during classification"}
    )
    labels_num: Optional[int] = field(
        default=2, metadata={"help": "number of labels in the model output"}
    )
    task1_labels_num: Optional[int] = field(
        default=4, metadata={"help": "number of labels in the model output of the first task"}
    )
    task2_labels_num: Optional[int] = field(
        default=3, metadata={"help": "number of labels in the model output of the second task"}
    )
    task3_labels_num: Optional[int] = field(
        default=4, metadata={"help": "number of labels in the model output of the third task"}
    )
    task4_labels_num: Optional[int] = field(
        default=3, metadata={"help": "number of labels in the model output of the fourth task"}
    )



@dataclass
class KfoldTrainingArguments(TrainingArguments):
    class_weights: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use class weights for imbalanced data."
        },
    )
    folds_num: Optional[int] = field(
        default=None,
        metadata={"help": "The number of folds."},
    )
    output_dir_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Output path prefix"}
    )
    project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Project name in wandb"}
    )


def parse_arguments():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, KfoldTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    for x in (model_args, data_args, training_args):
        pprint(x)
    return model_args, data_args, training_args
