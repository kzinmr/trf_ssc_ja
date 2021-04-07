import argparse
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_metric
from transformers import (
    Trainer,
    TrainingArguments,
)
from data import PAD_TOKEN_LABEL_ID
from model import BERTForSentenceSequenceClassification
from datamodule import SentenceSequenceClassificationDataModule
from tokenizer import custom_tokenizer_from_pretrained

metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != PAD_TOKEN_LABEL_ID]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != PAD_TOKEN_LABEL_ID]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def make_common_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--data_dir",
        default="/app/workspace/data",
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        default='/app/workspace/cache',
        type=str,
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        help="Path to pretrained model config (for transformers)",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        type=str,
        help="Path to pretrained tokenzier JSON config (for transformers)",
    )
    parser.add_argument(
        "--labels",
        default="workspace/data/label_types.txt",
        type=str,
        help="Path to a file containing all labels. (for transformers)",
    )
    return parser


def build_args(model_checkpoint=None):
    parser = make_common_args()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    # parser = TokenClassificationModule.add_model_specific_args(parent_parser=parser)
    parser = SentenceSequenceClassificationDataModule.add_model_specific_args(
        parent_parser=parser
    )
    args = parser.parse_args()
    # if notebook:
    #     args = parser.parse_args(args=[])
    pl.seed_everything(args.seed)

    if model_checkpoint:
        args.model_name_or_path = model_checkpoint

    args.gpu = torch.cuda.is_available()
    # args.num_samples = 20000

    args.delimiter = "\t"
    return args


if __name__ == "__main__":

    conll03 = False
    ja_gsd = False
    custom_pretrained = False
    data_dir = "/app/workspace/data"
    if not (Path(data_dir) / f"train.txt").exists():
        exit(0)

    # make dataset and tokenizer
    args = build_args()
    args.model_name_or_path = "cl-tohoku/bert-base-japanese-v2"

    model_checkpoint = args.model_name_or_path
    # args.tokenizer_path = "tokenizer.json"
    args.tokenizer_path = model_checkpoint

    dm = SentenceSequenceClassificationDataModule(args)
    dm.prepare_data()

    train_dataset = dm.train_dataset.to_dict()
    val_dataset = dm.val_dataset.to_dict()
    test_dataset = dm.test_dataset.to_dict()
    label_list = dm.label_list

    model = BERTForSentenceSequenceClassification( #.from_pretrained(
        model_checkpoint, num_labels=len(label_list)
    )
    tokenizer = custom_tokenizer_from_pretrained(args.tokenizer_path)

    args = TrainingArguments(
        "trf-ssc",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

    # prediction
    output = trainer.predict(test_dataset)
    print(output.metrics)

    input_ids = [d["input_ids"].tolist() for d in test_dataset]
    predictions = np.argmax(output.predictions, axis=2)
    labels = [d["labels"].tolist() for d in test_dataset]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != PAD_TOKEN_LABEL_ID]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != PAD_TOKEN_LABEL_ID]
        for prediction, label in zip(predictions, labels)
    ]
    input_sentences = [
        [
            tokenizer.convert_ids_to_tokens(ids)
            for (ids, l) in zip(sentence_ids, label)
            if l != PAD_TOKEN_LABEL_ID
        ]
        for sentence_ids, label in zip(input_ids, labels)
    ]
    with open(os.path.join(data_dir, "test.tsv"), "w") as fp:
        for sentences, golds, preds in zip(
            input_sentences, true_labels, true_predictions
        ):
            for sentence, g, p in zip(sentences, golds, preds):
                s = "_".join(sentence)
                fp.write(f"{s}\t{g}\t{p}\n")
            fp.write("\n")

    trainer.save_model(data_dir)