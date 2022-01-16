import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class EuropolisDatasetFeats(Dataset):
    """
    A Europolis Dataset. Takes a pandas dataframe and loads the features, labels and encodings given the tokenizer
    """

    def __init__(self, path_to_dataset, label, tokenizer, text_col):
        """

        :param path_to_dataset: the path to the dataframe, e.g. to the validation data
        :param label: the quality dimension to be predicted
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text
        """
        self.dataset = pd.read_csv(path_to_dataset, sep="\t")
        self.labels = self.dataset[label].values
        self.text_col = text_col
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]

        self.encodings = tokenizer(list(self.dataset[self.text_col]), return_tensors='pt', padding=True,
                                   truncation=True)

        self.num_cols = ["cogency", "effectiveness", "reasonableness", "overall"]
        self.feature_vectors = self.get_numerical_feats()

    def get_numerical_feats(self):
        # get a torch tensor of the argument quality scores
        return torch.tensor(self.dataset[self.num_cols].values.astype('float'), dtype=torch.float32)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['features'] = torch.tensor(self.feature_vectors[idx])
        return item

    def __len__(self):
        return len(self.labels)


class EuropolisDataset(Dataset):
    """
    A Europolis Dataset. Takes a pandas dataframe and loads the features, labels and encodings given the tokenizer
    """

    def __init__(self, path_to_dataset, label, tokenizer, text_col):
        """

        :param path_to_dataset: the path to the dataframe, e.g. to the validation data
        :param label: the quality dimension to be predicted
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text
        """
        self.dataset = pd.read_csv(path_to_dataset, sep="\t")
        self.labels = self.dataset[label].values
        self.text_col = text_col
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]

        self.encodings = tokenizer(list(self.dataset[self.text_col]), return_tensors='pt', padding=True,
                                   truncation=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class EuropolisSimpleDataset(Dataset):
    """
    A Europolis Dataset. Takes a pandas dataframe and loads the features, labels and encodings given the tokenizer
    """

    def __init__(self, path_to_dataset, label, text_col):
        """

        :param path_to_dataset: the path to the dataframe, e.g. to the validation data
        :param label: the quality dimension to be predicted
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text
        """
        self.dataset = pd.read_csv(path_to_dataset, sep="\t")
        self.labels = self.dataset[label].values
        self.text_col = text_col
        self.texts = self.dataset[text_col].values
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]
        self.num_cols = ["cogency", "effectiveness", "reasonableness", "overall"]
        self.feature_vectors = self.get_numerical_feats()

    def get_numerical_feats(self):
        # get a torch tensor of the argument quality scores
        return np.array(self.dataset[self.num_cols].values.astype('float'))

    def __getitem__(self, idx):
        item = {}
        item['text'] = self.texts[idx]
        item['features'] = self.feature_vectors[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class EuropolisFeatureDataset(Dataset):
    """
    A Europolis Dataset. Takes a pandas dataframe and loads the features, labels and encodings given the tokenizer
    """

    def __init__(self, path_to_dataset, label, text_col):
        """

        :param path_to_dataset: the path to the dataframe, e.g. to the validation data
        :param label: the quality dimension to be predicted
        :param tokenizer: a tokenizer object (e.g. RobertaTokenizer) that has a default for creating encodings for text
        :param text_col: the column name that stores the actual text
        """
        self.dataset = pd.read_csv(path_to_dataset, sep="\t")
        self.dataset = self.dataset.drop(columns=["filename_x", "filename_y"])
        self.labels = self.dataset[label].values
        self.text_col = text_col
        self.texts = self.dataset[text_col].values
        # drop all columns with no text
        self.dataset = self.dataset[self.dataset[self.text_col].notna()]
        self.num_cols = self.dataset.columns[29:]
        self.feature_vectors = self.get_numerical_feats()

    def get_numerical_feats(self):
        # get a torch tensor of the argument quality scores
        return np.array(self.dataset[self.num_cols].values.astype('float'))

    def __getitem__(self, idx):
        item = {}
        item['text'] = self.texts[idx]
        item['features'] = self.feature_vectors[idx]
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
