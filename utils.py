import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset


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
    def __init__(self):
        self.labels = set()

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  #.find(',')
                    ent2text[temp[0]] = temp[1]  #[:end]
  
        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    ent2text[temp[0]] = temp[1]  #[:first_sent_end_position + 1] 

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
                label = "1"
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label=label))
                
            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                text_a = head_ent_text
                text_b = relation_text
                text_c = tail_ent_text 
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = text_c, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    for j in range(5):
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
                            InputExample(guid=guid, text_a=tmp_head_text, text_b=text_b, text_c = text_c, label="0"))       
                else:
                    # corrupting tail
                    tmp_tail = ''
                    for j in range(5):
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
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c = tmp_tail_text, label="0"))                                                  
        return examples


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


class BertTrainDataset(Dataset):
    def __init__(self, triples, ent2input, rel2input, max_len, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.ent2input = ent2input
        self.rel2input = rel2input
        self.max_len = max_len

    def __convert_triple_to_bert_input(self, h, r, t):
        CLS, SEP = [101], [102]

        head = CLS + self.ent2input[h] + SEP
        seg_head = [0] * len(head)
        rel = self.rel2input[r] + SEP
        seg_rel = [1] * len(rel)
        tail = self.ent2input[t] + SEP
        seg_tail = [0] * len(tail)

        pos = head + rel + tail
        seg_pos = seg_head + seg_rel + seg_tail
        mask_pos = [1] * len(pos)

        padding = [0] * (self.max_len - len(pos))
        pos += padding
        seg_pos += padding
        mask_pos += padding

        return pos[:self.max_len], seg_pos[:self.max_len], mask_pos[:self.max_len]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        inputs, segment_ids, attn_masks = [], [], []
        labels = []
        head_ids, relation_ids, tail_ids = [], [], []
        head, relation, tail = self.triples[idx]  # true triple
        pos, seg_pos, mask_pos = self.__convert_triple_to_bert_input(head, relation, tail)
        inputs.append(pos)
        segment_ids.append(seg_pos)
        attn_masks.append(mask_pos)
        labels.append(1)
        head_ids.append(head)
        relation_ids.append(relation)
        tail_ids.append(tail)

        #subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        #subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        for neg in negative_sample:
            if self.mode == 'head-batch':
                ids, seg, mask = self.__convert_triple_to_bert_input(neg, relation, tail)
                head_ids.append(neg)
                relation_ids.append(relation)
                tail_ids.append(tail)
            elif self.mode == 'tail-batch':
                ids, seg, mask = self.__convert_triple_to_bert_input(head, relation, neg)
                head_ids.append(head)
                relation_ids.append(relation)
                tail_ids.append(neg)
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            inputs.append(ids)
            segment_ids.append(seg)
            attn_masks.append(mask)
            labels.append(0)

        inputs = torch.LongTensor(inputs)
        segment_ids = torch.LongTensor(segment_ids)
        attn_masks = torch.LongTensor(attn_masks)
        labels = torch.LongTensor(labels)
        head_ids = torch.LongTensor(head_ids)
        relation_ids = torch.LongTensor(relation_ids)
        tail_ids = torch.LongTensor(tail_ids)

        return inputs, segment_ids, attn_masks, labels, head_ids, relation_ids, tail_ids

    @staticmethod
    def collate_fn_bert(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.cat([_[3] for _ in data], dim=0)
        return inputs, segment_ids, attn_masks, labels

    @staticmethod
    def collate_fn_full(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.cat([_[3] for _ in data], dim=0)
        head_ids = torch.cat([_[4] for _ in data], dim=0)
        relation_ids = torch.cat([_[5] for _ in data], dim=0)
        tail_ids = torch.cat([_[6] for _ in data], dim=0)
        return inputs, segment_ids, attn_masks, labels, head_ids, relation_ids, tail_ids

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count
