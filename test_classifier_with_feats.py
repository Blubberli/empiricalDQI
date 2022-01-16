from seq_with_feats import RobertaWithFeats
from transformers import RobertaConfig, RobertaTokenizer
import numpy as np
import torch
import unittest
from transformers import TrainingArguments, Trainer
from data import EuropolisDataset
from torch.utils.data import DataLoader


class TestFeatureModels(unittest.TestCase):
    """
    this class tests the BasicTwoWordClassifier
    This test suite can be ran with:
        python -m unittest -q tests.BasicTwoWordClassifierTest
    """

    def setUp(self):
        # First, specify the ordinary BERT parameters by taking them from the
        # 'bert-base-uncased' model.
        # Also set the number of labels.
        config = RobertaConfig.from_pretrained(
            'roberta-base',
            num_labels=3,  # The number of output labels--3 for classification.
        )

        # Pass in the number of numerical features.
        config.numerical_feat_dim = 4

        # Pass in the size of the text embedding.
        # The text feature dimension is the "hidden_size" parameter which
        # comes from RobertaConfig. The length is 768 in ROBERTA-base (and most other BERT
        # models).
        config.text_feat_dim = config.hidden_size  # 768
        # load the adapted model with the modified config
        self._model = RobertaWithFeats(roberta_config=config)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.train_data = EuropolisDataset(path_to_dataset="data/5foldStratified/jcon/split0/train.csv",
                                           tokenizer=self.tokenizer, label='jcon', text_col='cleaned_comment')
        self.dev_data = EuropolisDataset(path_to_dataset="data/5foldStratified/jcon/split0/val.csv",
                                         tokenizer=self.tokenizer, label='jcon', text_col='cleaned_comment')
        self.test_data = EuropolisDataset(path_to_dataset="data/5foldStratified/resp_gr/split2/test.csv",
                                          tokenizer=self.tokenizer, label='resp_gr', text_col='cleaned_comment')

    def test_forward(self):
        """
        tests the classifier implemented in BasicTwoWordClassifier and the overridden method "forward"
        checks whether the output layer is of the right size
        """
        inputs = self.tokenizer(["In my opinion, the food is really special.", "I hate chips, they do not taste good."],
                                return_tensors="pt")
        # create a batch of 2
        labels = torch.tensor([[1], [1]])  # Batch size 2
        # create a random feature vector of length 5
        features = torch.from_numpy(np.random.rand(2, 4)).float()

        # call the forward method with the encoded input and the features
        outputs = self._model(**inputs, features=features, labels=labels)
        logits = outputs.logits
        expected_size = torch.tensor(np.zeros((2, 3))).shape
        np.testing.assert_allclose(logits.shape, expected_size)

    def test_dataset(self):
        """Test whether the dataset contains features and input ids and whether the features have the correct dimension"""
        dataloader = DataLoader(self.train_data, batch_size=3)
        batch = next(iter(dataloader))
        np.testing.assert_equal("features" in batch, True)
        np.testing.assert_equal("input_ids" in batch, True)
        np.testing.assert_equal(batch["features"].shape, [3, 4])

    def test_trainer(self):
        training_args = TrainingArguments("test_trainer")
        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.dev_data)
        result = trainer.predict(self.dev_data)
        print(result)
