import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lp_num_labels = config.lp_num_labels
        self.rp_num_labels = config.rp_num_labels
        self.lp_classifier = nn.Linear(config.hidden_size, self.lp_num_labels)
        self.rp_classifier = nn.Linear(config.hidden_size, self.rp_num_labels)
        self.mr_classifier = nn.Linear(config.hidden_size, 1)
        self.layer_norm = nn.LayerNorm(config.hidden_size, self.lp_num_labels)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, task=None,
                input_ids2=None, token_type_ids2=None, attention_mask2=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if task == "lp":
            logits = self.lp_classifier(pooled_output)
        elif task == "rp":
            logits = self.rp_classifier(pooled_output)
        elif task == "rr":
            logits1 = self.sigmoid(self.mr_classifier(pooled_output))
            outputs2 = self.bert(input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
            pooled_output2 = outputs2[1]
            pooled_output2 = self.dropout(pooled_output2)
            logits2 = self.sigmoid(self.mr_classifier(pooled_output2))
            return logits1, logits2
        else:
            raise TypeError

        return logits
