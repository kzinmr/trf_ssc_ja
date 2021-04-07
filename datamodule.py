import logging
import os
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BatchEncoding,
    PreTrainedTokenizerFast,
)

from data import *
from tokenizer import custom_tokenizer_from_pretrained

logger = logging.getLogger(__name__)

import numpy as np


class NoLocalFileError(Exception):
    pass


class ExamplesBuilder:
    def __init__(
        self,
        data_dir: str,
        split: Split,
        delimiter: str = "\t",
    ):
        """
        -
        """
        datadir_p = Path(data_dir)
        if (datadir_p / f"{split.value}.txt").exists():

            start = time.time()
            self.examples = self.read_examples_from_file(
                data_dir, split, delimiter=delimiter
            )
            end = time.time()
            read_time = end - start
            logger.info(f"READ TIME({split.value}): {read_time}")
        else:
            raise NoLocalFileError

    @staticmethod
    def is_boundary_line(line: str) -> bool:
        return line.startswith("-DOCSTART-") or line == "" or line == "\n"

    @staticmethod
    def read_examples_from_file(
        data_dir: str,
        mode: Union[Split, str],
        delimiter: str = "\t",
        label_idx: int = -1,
    ) -> List[SentenceSequenceClassificationExample]:
        """
        Read examples from file.
        - `line\tlabel` format, sentences are splitted by '\n\n'
        """
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        lines = []
        with open(file_path, encoding="utf-8") as f:
            lines = f.read().split("\n")

        guid_index = 1
        examples = []
        sentences = []
        labels = []
        for line in lines:
            if ExamplesBuilder.is_boundary_line(line):
                if sentences:
                    examples.append(
                        SentenceSequenceClassificationExample(
                            guid=f"{mode}-{guid_index}",
                            sentences=sentences,
                            labels=labels,
                        )
                    )
                    guid_index += 1
                    sentences = []
                    labels = []
            else:
                splits = line.strip().split(delimiter)
                sentences.append(splits[0])
                if len(splits) > 1:
                    label = splits[label_idx]
                    labels.append(BIOTag(label))
                else:
                    # for mode = "test"
                    labels.append(BIOTag("O"))
        if sentences:
            examples.append(
                SentenceSequenceClassificationExample(
                    guid=f"{mode}-{guid_index}", sentences=sentences, labels=labels
                )
            )
        return examples


class SentenceSequenceClassificationDataset(Dataset):
    """
    Build feature dataset so that the model can load
    """

    def __init__(
        self,
        examples: List[SentenceSequenceClassificationExample],
        tokenizer: PreTrainedTokenizerFast,
        label_to_id: Dict[str, int],
        tokens_per_sentence: int = 32,
        sentences_per_batch: int = 16,
        window_stride: int = -1,
    ):
        """tokenize_and_align_labels with long text (i.e. truncation is disabled)"""

        self.window_stride = sentences_per_batch
        if window_stride > 0 and window_stride < sentences_per_batch:
            self.window_stride = window_stride
        self.tokens_per_sentence = tokens_per_sentence
        self.sentences_per_batch = sentences_per_batch

        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

        # (n_windows, sentences_per_batch, tokens_per_sentence): dataloaderでバッチに分割
        self.features = self.make_window_features(examples)
        self._n_features = len(self.features)
        self.features_dict = self.to_dict()

    def make_window_features(
        self, examples: List[SentenceSequenceClassificationExample]
    ) -> List[BERTInputFeaturesBatch]:
        features: List[BERTInputFeaturesBatch] = []
        for example in examples:
            n_sentences = example.num_sentences
            sentences = example.sentences
            labels = [tag.label for tag in example.labels]
            for start in range(0, n_sentences, self.window_stride):
                end = min(start + self.sentences_per_batch, n_sentences)
                n_padding_to_add = max(0, self.sentences_per_batch - end + start)
                sentences_window = sentences[start:end]
                labels_window = labels[start:end]

                # (sentences_per_batch, seq_len)
                sentences_enc: BatchEncoding = self.tokenizer(
                    sentences_window,
                    truncation=True,
                    padding='max_length',
                    add_special_tokens=True,
                    max_length=self.tokens_per_sentence,
                )

                label_ids_nopad = [self.label_to_id[label] for label in labels_window]
                # token-wise padding
                # sentence-wise padding
                input_ids_nopad = sentences_enc["input_ids"]
                attention_mask_nopad = sentences_enc["attention_mask"]
                input_padding = [
                    [
                        self.tokenizer.pad_token_id
                        for _ in range(self.tokens_per_sentence)
                    ]
                    for _ in range(n_padding_to_add)
                ]
                attention_mask_padding = [
                    [0 for _ in range(self.tokens_per_sentence)]
                    for _ in range(n_padding_to_add)
                ]
                labels_padding = [PAD_TOKEN_LABEL_ID] * n_padding_to_add

                input_ids = torch.LongTensor(input_ids_nopad + input_padding)
                attention_mask = torch.LongTensor(
                    attention_mask_nopad + attention_mask_padding
                )
                label_ids = torch.LongTensor(label_ids_nopad + labels_padding)

                bert_batch = BERTInputFeaturesBatch(
                    input_ids=input_ids, attention_mask=attention_mask, labels=label_ids
                )

                features.append(bert_batch)
        return features

    def __len__(self):
        return self._n_features

    def __getitem__(self, idx) -> BERTInputFeaturesBatch:
        return self.features[idx]

    def to_dict(self):
        return [
            {
                "input_ids": feature.input_ids,
                "attention_mask": feature.attention_mask,
                "labels": feature.labels,
            }
            for feature in self.features
        ]


class SentenceSequenceClassificationDataModule(pl.LightningDataModule):
    """
    Prepare dataset and build DataLoader
    """

    def __init__(self, hparams: Namespace):
        self.tokenizer: PreTrainedTokenizerFast
        self.train_dataset: SentenceSequenceClassificationDataset
        self.val_dataset: SentenceSequenceClassificationDataset
        self.test_dataset: SentenceSequenceClassificationDataset
        # self.label_token_aligner: LabelTokenAligner
        # self.train_examples: List[StringSpanExample]
        # self.val_examples: List[StringSpanExample]
        # self.test_examples: List[StringSpanExample]

        super().__init__()

        self.cache_dir = hparams.cache_dir if hparams.cache_dir else None
        if self.cache_dir is not None and not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        self.data_dir = hparams.data_dir
        # if not os.path.exists(self.data_dir):
        #     os.mkdir(self.data_dir)

        self.tokenizer_name_or_path = (
            hparams.tokenizer_path
            if hparams.tokenizer_path
            else hparams.model_name_or_path
        )
        self.labels_path = hparams.labels

        # batch sizer for LSTM+Linear
        self.train_batch_size = hparams.train_batch_size
        self.eval_batch_size = hparams.eval_batch_size
        self.num_workers = hparams.num_workers
        self.num_samples = hparams.num_samples

        # this become batch-size for BERT
        self.tokens_per_sentence = hparams.tokens_per_sentence
        self.sentences_per_batch = hparams.sentences_per_batch
        self.window_stride = hparams.window_stride

        self.delimiter = hparams.delimiter

    def prepare_data(self):
        """
        Downloads the data and prepare the tokenizer
        """
        self.tokenizer = custom_tokenizer_from_pretrained(
            self.tokenizer_name_or_path, self.cache_dir
        )
        try:
            self.train_examples = ExamplesBuilder(
                self.data_dir,
                Split.train,
                delimiter=self.delimiter,
            ).examples
            self.val_examples = ExamplesBuilder(
                self.data_dir,
                Split.dev,
                delimiter=self.delimiter,
            ).examples
            self.test_examples = ExamplesBuilder(
                self.data_dir,
                Split.test,
                delimiter=self.delimiter,
            ).examples

            if self.num_samples > 0:
                self.train_examples = self.train_examples[: self.num_samples]
                self.val_examples = self.val_examples[: self.num_samples]
                self.test_examples = self.test_examples[: self.num_samples]

            # create label vocabulary from dataset
            all_examples = self.train_examples + self.val_examples + self.test_examples
            all_labels = sorted(
                {
                    tag.label
                    for ex in all_examples
                    for tag in ex.labels
                    if tag.bio != BIO.O
                }
            )
            self.label_list = [BIO.O.value] + sorted(all_labels)
            label_types = sorted(
                {
                    tag.tagtype.value
                    for ex in all_examples
                    for tag in ex.labels
                    if tag.bio != BIO.O
                }
            )
            with open(self.labels_path, "w") as fp:
                for l in label_types:
                    fp.write(l)
                    fp.write("\n")

            self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
            self.id_to_label = self.label_list

            start = time.time()
            self.train_dataset = self.create_dataset(
                self.train_examples, self.tokenizer, self.label_to_id
            )
            end = time.time()
            read_time = end - start
            logger.info(f"DATASET TIME(train): {read_time}")

            start = time.time()
            self.val_dataset = self.create_dataset(
                self.val_examples, self.tokenizer, self.label_to_id
            )
            end = time.time()
            read_time = end - start
            logger.info(f"DATASET TIME(val): {read_time}")

            start = time.time()
            self.test_dataset = self.create_dataset(
                self.test_examples, self.tokenizer, self.label_to_id
            )
            end = time.time()
            read_time = end - start
            logger.info(f"DATASET TIME(test): {read_time}")

            self.dataset_size = len(self.train_dataset)

            logger.info(self.val_examples[:3])
            logger.info(self.val_dataset[:3])

        except NoLocalFileError as e:
            logger.error(e)
            exit(1)

    def setup(self, stage=None):
        """
        split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        # our dataset is splitted in prior

    def create_dataset(
        self,
        examples: List[SentenceSequenceClassificationExample],
        tokenizer,
        label_to_id,
    ) -> SentenceSequenceClassificationDataset:
        return SentenceSequenceClassificationDataset(
            examples,
            tokenizer,
            label_to_id,
            tokens_per_sentence=self.tokens_per_sentence,
            sentences_per_batch=self.sentences_per_batch,
            window_stride=self.window_stride,
        )

    @staticmethod
    def create_dataloader(
        ds: SentenceSequenceClassificationDataset,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = False,
    ) -> DataLoader:

        return DataLoader(
            ds.to_dict(),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.create_dataloader(
            self.train_dataset,
            self.train_batch_size,
            self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.create_dataloader(
            self.val_dataset,
            self.eval_batch_size,
            self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.create_dataloader(
            self.test_dataset,
            self.eval_batch_size,
            self.num_workers,
            shuffle=False,
        )

    def total_steps(self) -> int:
        """
        The number of total training steps that will be run. Used for lr scheduler purposes.
        """
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.accumulate_grad_batches
            * num_devices
        )
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=4,
            help="input batch size for training (default: 32)",
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=4,
            help="input batch size for validation/test (default: 32)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            metavar="N",
            help="number of workers (default: 3)",
        )
        parser.add_argument(
            "--tokens_per_sentence",
            default=32,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--sentences_per_batch",
            default=8,
            type=int,
            help="The maximum number of sentences. This is the batch size for BERT input"
            "and also the maximum number of sequences for LSTM sequence tagger.",
        )
        parser.add_argument(
            "--window_stride",
            default=-1,
            type=int,
            help="The stride of moving window over input sencence lines."
            "This must be shorter than sentences_per_batch.",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=-1,
            metavar="N",
            help="Number of samples to be used for training and evaluation steps",
        )
        parser.add_argument(
            "--delimiter",
            default="\t",
            type=str,
            help="delimiter between sentence and label in one line.",
        )

        return parser
