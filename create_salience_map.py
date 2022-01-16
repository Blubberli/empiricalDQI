from transformers_interpret import SequenceClassificationExplainer
import transformers
# import nlp
import torch

from transformers import set_seed
from transformers import AutoTokenizer
import pandas as pd
from scipy.special import softmax
import argparse
from tqdm import tqdm

set_seed(42)

from interpret_with_transformer import load_model, class_name_dic


def create_map(sentence, dimension, output_path, model, tokenizer):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer,
                                                    custom_labels=class_name_dic[dimension])
    cls_explainer(sentence)
    cls_explainer.visualize(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='path to the model')
    parser.add_argument("label_num", type=int)
    parser.add_argument("dimension", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("sentence", type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', padding=True, max_length=512, truncation=True)
    tokenizer.padding = True
    tokenizer.max_len = 512
    tokenizer.max_length = 512
    tokenizer.truncation = True
    model = load_model(args.model, args.label_num)
    create_map(sentence=args.sentence, model=model, tokenizer=tokenizer, output_path=args.outfile,
               dimension=args.dimension)
