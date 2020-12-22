import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lp_num_labels = config.lp_n_label
        self.rp_num_labels = config.rp_n_label
        self.lp_classifier = nn.Linear(config.hidden_size, self.lp_num_labels)
        self.rp_classifier = nn.Linear(config.hidden_size, self.rp_num_labels)
        self.mr_classifier = nn.Linear(config.hidden_size, 1)
        self.layer_norm = nn.LayerNorm(config.hidden_size, self.lp_num_labels)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_bert_weights)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            task=None,
    ):
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if task == "lp":
            logits = self.lp_classifier(pooled_output)
        elif task == "rp":
            logits = self.rp_classifier(pooled_output)
        elif task == "mr":
            logits = self.sigmoid(self.mr_classifier(pooled_output))
        else:
            raise TypeError

        return logits
