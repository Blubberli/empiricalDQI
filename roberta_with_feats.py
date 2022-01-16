from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch
from torch import nn


class RobertaWithFeats(RobertaForSequenceClassification):
    """
    A model for classification or regression which combines text and numerical features.
    The text features are processed with Roberta. All
    features are concatenated into a single vector, which is fed into the RobertaClassification Head
    for final classification / regression.

    This class expects a transformers.RobertaConfig object, and the config object
    needs to have two additional properties manually added to it:
      `text_feat_dim` - The length of the BERT vector.
      `numerical_feat_dim` - The number of numerical features.
    """

    def __init__(self, roberta_config):
        # Call the constructor for the huggingface `RobertaForSequenceClassification`
        # class, which will do all of the BERT-related setup. The resulting ROBERTA
        # model is stored in `self.roberta`.
        super().__init__(roberta_config)

        # Store the number of labels, which tells us whether this is a
        # classification or regression task.
        self.num_labels = roberta_config.num_labels
        # Calculate the combined vector length.
        combined_feat_dim = roberta_config.text_feat_dim + \
                            roberta_config.numerical_feat_dim
        # Create a batch normalizer for the numerical features.
        self.num_bn = nn.BatchNorm1d(roberta_config.numerical_feat_dim)
        # set the hidden size for the classifier to the combined_feat_dim
        roberta_config.hidden_size = combined_feat_dim
        self.classifier = FeatureAdaptedClassificationHead(roberta_config)
        self.config = roberta_config

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            features=None
    ):
        r"""
        perform a forward pass of our model.

        This has the same inputs as `forward` in `RobertaForSequenceClassification`,
        but with one extra parameter:
          `features` - Tensor of numerical features.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Run the text through the ROBERTA model. Invoking `self.roberta` returns
        # outputs from the encoding layers, and not from the final classifier.
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # take <s> token (equiv. to [CLS])
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 0, :]
        # Apply batch normalization to the numerical features.
        numerical_feats = self.num_bn(features)
        # Simply concatenate everything into one vector.
        # For example, if we have 3 numer. features, then the
        # result has 768 + 3 = 771 features

        combined_feats = torch.cat((sequence_output, numerical_feats),
                                   dim=1)
        logits = self.classifier(combined_feats)

        loss = None
        # compute the loss as it is done in the original robertaforseqclassification code
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        # return the result as it is done in the original code, loss, logits, hidden states and attentions
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FeatureAdaptedClassificationHead(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        # call the constructor of the original classification head, which is a non-linear transformation with
        # a linear output layer and some dropout
        super().__init__(config)

    def forward(self, combined_feature_vector, **kwargs):
        x = combined_feature_vector  # this is the concatenation of features and roberta CLS embedding
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
