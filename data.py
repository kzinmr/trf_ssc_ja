import torch
from dataclasses import dataclass, field
from enum import Enum
from typing import List

IntList = List[int]
IntListList = List[IntList]
StrList = List[str]
StrListList = List[StrList]
PAD_TOKEN_LABEL_ID = -100
PAD_TOKEN = "[PAD]"


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class BIO(Enum):
    B = "B"
    I = "I"
    O = "O"


class TagType(Enum):
    Other = "Other"
    level1 = "level1"
    level2 = "level2"
    level3 = "level3"


@dataclass
class BIOTag:
    _label: str
    bio: BIO = field(init=False)
    tagtype: TagType = field(init=False)
    label: str = field(init=False)

    def __post_init__(self):
        label = self._label
        if len(label.split("-")) == 2:
            bio_repr, tagtype_repr = label.split("-")
            try:
                self.bio = BIO(bio_repr)
                self.tagtype = TagType(tagtype_repr)
            except ValueError:
                self.bio = BIO.O
                self.tagtype = TagType.Other
        else:
            self.bio = BIO.O
            self.tagtype = TagType.Other
        self.label = (
            f"{self.bio.value}-{self.tagtype.value}" if self.bio != BIO.O else "O"
        )

    @staticmethod
    def from_values(bio: BIO, tag_type: TagType):
        if bio != BIO.O:
            return BIOTag(f"{bio.value}-{tag_type.value}")
        else:
            return BIOTag("O")


@dataclass
class SentenceSequenceClassificationExample:
    guid: str
    sentences: StrList
    labels: List[BIOTag]

    def __post_init__(self):
        assert len(self.sentences) == len(self.labels)

    @property
    def num_sentences(self):
        return len(self.sentences)


@dataclass
class BERTInputFeaturesBatch:
    # (sentences_per_batch, max_seq_len) tensor
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor

    def __post_init__(self):
        assert self.input_ids.shape == self.attention_mask.shape
        assert self.input_ids.shape[0] == self.labels.shape[0]
