import torch
import wandb
from pathlib import Path
from utils import get_name_with_hyperparams, reset_wandb_env
from args import parse_arguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import set_seed
from transformers.integrations import WandbCallback
import torch.nn.functional as F
from roberta_with_feats import RobertaWithFeats
from transformers import RobertaConfig, EarlyStoppingCallback
from data import EuropolisDataset, EuropolisDatasetFeats
import numpy as np
from evaluation import average_all, average_class
import os

os.environ["WANDB_START_METHOD"] = "thread"
# set seed to 42 for reproducibility
set_seed(42)

# check for GPUs or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU in use:')
else:
    print('using the CPU')
    device = torch.device("cpu")


def run_train_with_trainer(train_data, dev_data, test_data, data_args, model_args, training_args,
                           fold_id):
    # initialize classification model
    # general = name with important hyperparams
    # split_run_name = adding the split ID to the name
    general_name, split_run_name = get_name_with_hyperparams(data_args=data_args, model_args=model_args,
                                                             training_args=training_args,
                                                             fold_id=fold_id)
    # create a directory to store the results
    general_dir = training_args.output_dir
    split_dir = Path(training_args.output_dir + "/" + split_run_name)
    if not split_dir.exists():
        split_dir.mkdir()
    # start wanDB
    reset_wandb_env()
    wandb_run = wandb.init(project=training_args.project_name,
                           group=general_name,
                           name=split_run_name,
                           reinit=True,
                           settings=wandb.Settings(start_method="fork"))
    # create WanDB callback
    wandb_callback = WandbCallback()
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    # create early stopping callback
    print("number of labels: %d" % model_args.labels_num)

    # set the output directory to the run-specific directory to store the models there
    training_args.output_dir = split_dir

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        compute_metrics=compute_metrics,
        callbacks=[wandb_callback, early_stopping_callback]
    )

    # train with trainer
    trainer.train()

    train_results = trainer.evaluate(train_data)

    # evaluate on dev set
    dev_results = trainer.evaluate(dev_data)
    # evaluate on test set
    test_results = trainer.evaluate(test_data)

    dev_report = dev_results["eval_report_dict"]
    test_report = test_results["eval_report_dict"]

    wandb_run.finish()
    dev_predictions = trainer.predict(dev_data)
    test_predictions = trainer.predict(test_data)

    # generate probabilities over classes and save the test data with predictions as a dataframe into split directory
    dev_data.dataset['predictions'] = F.softmax(torch.tensor(dev_predictions.predictions), dim=-1).tolist()
    dev_data.dataset.to_csv(f'{str(split_dir)}/dev_df_with_predictions.csv', index=False, sep="\t")
    test_data.dataset['predictions'] = F.softmax(torch.tensor(test_predictions.predictions), dim=-1).tolist()
    test_data.dataset.to_csv(f'{str(split_dir)}/test_df_with_predictions.csv', index=False, sep="\t")
    # save classification report for training,  validation and test set in split directory
    with open(f'{str(split_dir)}/train_report.csv', "w") as f:
        f.write(train_results["eval_report_csv"])
    with open(f'{str(split_dir)}/dev_report.csv', "w") as f:
        f.write(dev_results["eval_report_csv"])
    with open(f'{str(split_dir)}/test_report.csv', "w") as f:
        f.write(test_results["eval_report_csv"])

    training_args.output_dir = general_dir

    return dev_report, test_report


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1).flatten()
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='macro')
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    report_dict = classification_report(y_true=labels, y_pred=preds, output_dict=True)
    report_csv = classification_report(y_true=labels, y_pred=preds)
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        "report_dict": report_dict,
        "report_csv": report_csv
    }
    return results


def save_results(output_dir, report, filename):
    with open(output_dir + "/%s" % filename, "w") as f:
        f.write(str(report))
    print("results saved in %s" % filename)


def get_datasets(use_feats, data_args, i):
    if use_feats:
        # create a training, dev and test dataset
        train = EuropolisDatasetFeats(path_to_dataset=data_args.data_dir + "/split%i/train.csv" % i,
                                      label=data_args.quality_dim, tokenizer=tokenizer, text_col=data_args.text_col)
        dev = EuropolisDatasetFeats(path_to_dataset=data_args.data_dir + "/split%i/val.csv" % i,
                                    label=data_args.quality_dim, tokenizer=tokenizer, text_col=data_args.text_col)
        test = EuropolisDatasetFeats(path_to_dataset=data_args.data_dir + "/split%i/test.csv" % i,
                                     label=data_args.quality_dim, tokenizer=tokenizer, text_col=data_args.text_col)
    else:
        train = EuropolisDataset(path_to_dataset=data_args.data_dir + "/split%i/train.csv" % i,
                                 label=data_args.quality_dim, tokenizer=tokenizer, text_col=data_args.text_col)
        dev = EuropolisDataset(path_to_dataset=data_args.data_dir + "/split%i/val.csv" % i,
                               label=data_args.quality_dim, tokenizer=tokenizer, text_col=data_args.text_col)
        test = EuropolisDataset(path_to_dataset=data_args.data_dir + "/split%i/test.csv" % i,
                                label=data_args.quality_dim, tokenizer=tokenizer, text_col=data_args.text_col)
    return train, dev, test


if __name__ == '__main__':
    # read in arguments
    # model args: all classification details
    # data args: path to dataset etc.
    # training args: learning rate, optimizer etc.
    model_args, data_args, training_args = parse_arguments()
    print("model is trained for %s " % data_args.quality_dim)
    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if model_args.use_feats:
        config = RobertaConfig.from_pretrained(
            'roberta-base',
            num_labels=model_args.labels_num,  # The number of output labels--3 for classification.
        )
        # Pass in the number of numerical features.
        config.numerical_feat_dim = 4

        # Pass in the size of the text embedding.
        # The text feature dimension is the "hidden_size" parameter which
        # comes from RobertaConfig. The length is 768 in ROBERTA-base (and most other BERT
        # models).
        config.text_feat_dim = config.hidden_size  # 768
        # load the adapted model with the modified config
        model = RobertaWithFeats(roberta_config=config)
    else:
        # use a normal seq classification model without features
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                                   num_labels=model_args.labels_num)
    print("loaded model of type %s" % str(type(model)))
    dev_reports = []
    test_reports = []
    for i in range(0, 5):
        train, dev, test = get_datasets(use_feats=model_args.use_feats, data_args=data_args, i=i)
        print(len(set(train.labels)))
        print(train.labels)
        dev_results, test_results = run_train_with_trainer(train_data=train,
                                                           dev_data=dev,
                                                           test_data=test,
                                                           data_args=data_args,
                                                           model_args=model_args,
                                                           training_args=training_args,
                                                           fold_id=str(i))
        dev_reports.append(dev_results)
        test_reports.append(test_results)
    # get average results for the 5 splits
    average_dev = average_all(dev_reports)
    average_test = average_all(test_reports)
    label_list = list(set(train.labels))
    class_average_dev = average_class(dev_reports, label_list=label_list)
    class_average_test = average_class(test_reports, label_list=label_list)
    # write them to file in the global dir
    with open(training_args.output_dir + "/average_dev.csv", "w") as f:
        f.write(average_dev)
    with open(training_args.output_dir + "/average_test.csv", "w") as f:
        f.write(average_test)
    with open(training_args.output_dir + "/average_class_dev.csv", "w") as f:
        f.write(class_average_dev)
    with open(training_args.output_dir + "/average_class_test.csv", "w") as f:
        f.write(class_average_test)
