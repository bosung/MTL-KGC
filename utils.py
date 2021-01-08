import csv
import sys
import os
import numpy as np
import random
import torch
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def order_selection(tasks, number_of_batches):
    order = []
    picked_num = {task: 0 for task in tasks}

    while(all(number_of_batches[task] == 0 for task in number_of_batches) is not True):
        total = sum([number_of_batches[tasks] for tasks in number_of_batches])
        prob = [number_of_batches[task]/total for task in number_of_batches]
        pick = np.random.choice(tasks, p=prob)

        order.append((pick, picked_num[pick]))
        number_of_batches[pick] -= 1
        picked_num[pick] += 1

    return order


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "lp":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class KGProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _create_examples(self, data):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        
    @classmethod
    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities
        
    @classmethod
    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    @classmethod
    def get_entity2text(self, data_dir):
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    ent2text[temp[0]] = temp[1]  # [:end]
        return ent2text

    @classmethod
    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    @classmethod
    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    @classmethod
    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class LPProcessor(KGProcessor):
    """Processor for the Link Prediction task."""
    def __init__(self, data_dir):
        self.labels = set()
        self._set_entities(data_dir)
        self._set_relations(data_dir)

    def _set_entities(self, data_dir):
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            entities = []
            ent2id = {}
            for i, line in enumerate(f.readlines()):
                ent = line.strip()
                entities.append(ent)
                ent2id[ent] = i

        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    # first_sent_end_position = temp[1].find(".")
                    if temp[0] not in ent2id:
                        i += 1
                        ent2id[temp[0]] = i

            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    # first_sent_end_position = temp[1].find(".")
                    if temp[0] not in ent2id:
                        i += 1
                        ent2id[temp[0]] = i

        self.entities = entities
        self.num_entity = len(entities)
        self.ent2id = ent2id

    def _set_relations(self, data_dir):
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            relations = []
            rel2id = {}
            for i, line in enumerate(f.readlines()):
                rel = line.strip()
                relations.append(rel)
                rel2id[rel] = i
        self.relations = relations
        self.num_relation = len(relations)
        self.rel2id = rel2id

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir, entity=False):
        """Gets training triples."""
        if entity:
            return self._get_triples(os.path.join(data_dir, "train.tsv"), entity=True)
        else:
            return self._get_triples(os.path.join(data_dir, "train.tsv"))

    def _get_triples(self, file_path, entity=False):
        triples = []
        with open(file_path) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                if entity:
                    triples.append([h, r, t])
                else:
                    triples.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
        return triples

    def get_entity2text(self, data_dir):
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    ent2text[temp[0]] = temp[1]  # [:end]

        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    # first_sent_end_position = temp[1].find(".")
                    #if temp[0] not in ent2text:
                    ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]
        return ent2text

    def get_relation2text(self, data_dir):
        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    rel2text[temp[0]] = temp[1]  # [:end]
        return rel2text

    def get_ent2input(self, data_dir, tokenizer, ent_max_len=128):
        ent2text = self.get_entity2text(data_dir)
        ent2input = {}
        for e in ent2text:
            e_id = self.ent2id[e]
            text = ent2text[e]
            tokens = tokenizer.tokenize(text)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            ent2input[e_id] = input_ids[:ent_max_len]
        return ent2input  # for max_len

    def get_rel2input(self, data_dir, tokenizer):
        rel2text = self.get_relation2text(data_dir)
        rel2input = {}
        for rel in rel2text:
            r_id = self.rel2id[rel]
            text = rel
            tokens = tokenizer.tokenize(text)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            rel2input[r_id] = input_ids
        return rel2input



def lp_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """"""
    """Loads a data file into a list of `InputBatch`s for the Link Prediction task.
       ex) the triple <plant tissue, _hypernym, plant structure> should be converted to
       "[CLS] plant tissue, the tissue of a plant [SEP] hypernym [SEP] plant structure, \\
        any part of a plant or fungus [SEP]"
    """

    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        tokens_c = None

        if example.text_b and example.text_c:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_c = tokenizer.tokenize(example.text_c)
            _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        if tokens_c:
            tokens += tokens_c + ["[SEP]"]
            segment_ids += [0] * (len(tokens_c) + 1)        

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


class RPProcessor(KGProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()

    def get_labels(self, data_dir):
        return self.get_relations(data_dir)
    
    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                ent2text[temp[0]] = temp[1]

        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    #first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1]#[:first_sent_end_position + 1]              

        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ent2text[line[0]]
            text_b = ent2text[line[2]]
            label = line[1]
            self.labels.add(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def rp_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


class RRProcessor(KGProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    ent2text[temp[0]] = temp[1]

        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(lines):

            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":
                # triple_label = line[3]
                # if triple_label == "1":
                #     label = "1"
                # else:
                #     label = "0"
                label = "1"

                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a1=text_a, text_b1=text_b, text_c1=text_c, label=label))

            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text
                # examples.append(
                #     InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    tmp_head = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[0])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_head = random.choice(tmp_ent_list)
                        tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_head_text = ent2text[tmp_head]
                    examples.append(
                        InputExample(guid=guid, text_a1=text_a, text_b1=text_b, text_c1=text_c,
                                     text_a2=tmp_head_text, text_b2=text_b, text_c2=text_c, label="1"))
                else:
                    # corrupting tail
                    tmp_tail = ''
                    while True:
                        tmp_ent_list = set(entities)
                        tmp_ent_list.remove(line[2])
                        tmp_ent_list = list(tmp_ent_list)
                        tmp_tail = random.choice(tmp_ent_list)
                        tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                        if tmp_triple_str not in lines_str_set:
                            break
                    tmp_tail_text = ent2text[tmp_tail]
                    examples.append(
                        InputExample(guid=guid, text_a1=text_a, text_b1=text_b, text_c1=text_c,
                                     text_a2=text_a, text_b2=text_b, text_c2=tmp_tail_text, label="1"))
        return examples


def rr_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info=True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a1 = tokenizer.tokenize(example.text_a1)
        tokens_b1 = tokenizer.tokenize(example.text_b1)
        tokens_c1 = tokenizer.tokenize(example.text_c1)
        _truncate_seq_triple(tokens_a1, tokens_b1, tokens_c1, max_seq_length - 4)

        if example.text_a2 and example.text_b2 and example.text_c2:
            tokens_a2 = tokenizer.tokenize(example.text_a2)
            tokens_b2 = tokenizer.tokenize(example.text_b2)
            tokens_c2 = tokenizer.tokenize(example.text_c2)
            _truncate_seq_triple(tokens_a2, tokens_b2, tokens_c2, max_seq_length - 4)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens1 = ["[CLS]"] + tokens_a1 + ["[SEP]"]
        segment_ids1 = [0] * len(tokens1)
        tokens1 += tokens_b1 + ["[SEP]"]
        segment_ids1 += [1] * (len(tokens_b1) + 1)
        tokens1 += tokens_c1 + ["[SEP]"]
        segment_ids1 += [0] * (len(tokens_c1) + 1)
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        input_mask1 = [1] * len(input_ids1)

        # Zero-pad up to the sequence length.
        padding1 = [0] * (max_seq_length - len(input_ids1))
        input_ids1 += padding1
        input_mask1 += padding1
        segment_ids1 += padding1

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length

        input_ids2 = None
        input_mask2 = None
        segment_ids2 = None

        if example.text_a2 and example.text_b2 and example.text_c2:
            tokens2 = ["[CLS]"] + tokens_a2 + ["[SEP]"]
            segment_ids2 = [0] * len(tokens2)
            tokens2 += tokens_b2 + ["[SEP]"]
            segment_ids2 += [1] * (len(tokens_b2) + 1)
            tokens2 += tokens_c2 + ["[SEP]"]
            segment_ids2 += [0] * (len(tokens_c2) + 1)
            input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
            input_mask2 = [1] * len(input_ids2)

            # Zero-pad up to the sequence length.
            padding2 = [0] * (max_seq_length - len(input_ids2))
            input_ids2 += padding2
            input_mask2 += padding2
            segment_ids2 += padding2

            assert len(input_ids2) == max_seq_length
            assert len(input_mask2) == max_seq_length
            assert len(segment_ids2) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens (True): %s" % " ".join([str(x) for x in tokens1]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids1]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask1]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids1]))

            if example.text_a2 and example.text_b2 and example.text_c2:
                logger.info("tokens (False): %s" % " ".join([str(x) for x in tokens2]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids2]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask2]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids2]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids1=input_ids1,
                          input_mask1=input_mask1,
                          segment_ids1=segment_ids1,
                          input_ids2=input_ids2,
                          input_mask2=input_mask2,
                          segment_ids2=segment_ids2, label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
