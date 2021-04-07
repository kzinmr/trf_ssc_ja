from typing import Dict, List
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel


class BERTForSentenceSequenceClassification(nn.Module):
    def __init__(self, checkpoint: str, num_labels: int):
        super(BERTForSentenceSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        ### New layers:
        lstm_hidden_size = 256
        hidden_dropout_prob = 0.5
        self.num_labels = num_labels
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.linear = nn.Linear(lstm_hidden_size * 2, self.num_labels)

    def apply_bert(self, input_ids, attention_mask):
        """Apply BertModel for inputs and return pooled_output."""
        return self.bert(input_ids, attention_mask)[1]

    def apply_bert_for_sentences(self, input_ids, attention_mask, axis=1):
        """Apply BERT for each sentences:
        - inputs: (batch_size, sentences_per_batch, tokens_per_sentences)
        - outputs: (batch_size, sentences_per_batch, hidden_size)
        """
        return torch.stack(
            [
                self.apply_bert(ids_sentence, mask_sentence)
                for ids_sentence, mask_sentence in zip(
                    torch.unbind(input_ids, dim=axis),
                    torch.unbind(attention_mask, dim=axis),
                )
            ],
            dim=axis,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.IntTensor = None,
    ):
        """Each input tensor corresponds to line windows in each documents.
        - input_ids: (batch_size, sentences_per_batch, tokens_per_sentences)
        - labels: (batch_size, sentences_per_batch)
        """
        lengths = [ids.shape[0] for ids in input_ids]  # sentences_per_batch
        # (batch_size, sentences_per_batch, hidden_size)
        sentence_tensor = self.apply_bert_for_sentences(
            input_ids, attention_mask, axis=1
        )

        # is this necessary for already padded tensors?
        packed = pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )
        # PackedSequence -> PackedSequence
        lstm_output, hidden = self.lstm(packed)
        outputs, output_lengths = pad_packed_sequence(lstm_output, batch_first=True)
        # (batch_size, sentences_per_batch, 2*lstm_hidden_size)
        outputs = self.dropout(outputs)
        # (batch_size, sentences_per_batch, num_labels)
        logits = self.linear(outputs)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return (loss, logits)
        else:
            return (logits,)
