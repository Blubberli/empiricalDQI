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

class_name_dic = {
    "jlev": ["no justification", "inferior", "qualitifed",
             "sophisticated"],
    "resp_gr": ["disrespectful", "implicit", "explicit"],
    "int1": ["no reference", "negative", "neutral", "positive"],

    "jcon": ["own country", "no reference", "common good"]
}


def load_model(model_path, label_num):
    model_config = transformers.AutoConfig.from_pretrained(
        'roberta-base',
        num_labels=label_num,
        output_hidden_states=True,
        output_attentions=True,
    )
    model = _from_pretrained(
        transformers.AutoModelForSequenceClassification,
        'roberta-base',
        config=model_config)
    print("model initialized")
    checkpoint = torch.load(model_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)
    print("loaded model")
    return model


def _from_pretrained(cls, *args, **kw):
    """Load a transformers model in PyTorch, with fallback to TF2/Keras weights."""
    try:
        return cls.from_pretrained(*args, **kw)
    except OSError as e:

        return cls.from_pretrained(*args, from_tf=True, **kw)


def get_attributions(path_to_data, model, tokenizer, dimension, output_path):
    """create a file with word attributions for a given test set"""
    data = pd.read_csv(path_to_data, sep="\t")
    comments = data.cleaned_comment.values
    true_labels = data[dimension].values
    cls_explainer = SequenceClassificationExplainer(model, tokenizer,
                                                    custom_labels=class_name_dic[dimension])

    with open(output_path, "w") as out:
        for i in tqdm(range(len(comments))):
            true_class = true_labels[i]
            comment = comments[i]
            s = comment + "\t" + "\t" + str(true_class) + "\t"
            word_attributions = cls_explainer(comment)
            s += cls_explainer.predicted_class_name + "\t" + str(word_attributions) + "\n"
            out.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='path to the test set to calculate attributions')
    parser.add_argument('model', type=str,
                        help='path to the model')
    parser.add_argument("dimension", type=str, help="the quality dimension")
    parser.add_argument("label_num", type=int)
    parser.add_argument("outfile", type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', padding=True, max_length=512, truncation=True)
    tokenizer.padding = True
    tokenizer.max_len = 512
    tokenizer.max_length = 512
    tokenizer.truncation = True
    model = load_model(args.model, args.label_num)
    get_attributions(args.dataset, model, tokenizer, args.dimension, args.outfile)
