import numpy as np
import torch
from torch.utils.data import Dataset


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


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
        CLS, SEP = [101], [102]  # for BERT index

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
        head, relation, tail = self.triples[idx]  # true triple
        pos, seg_pos, mask_pos = self.__convert_triple_to_bert_input(head, relation, tail)
        inputs = [pos]
        segment_ids = [seg_pos]
        attn_masks = [mask_pos]
        labels = [1]
        head_ids = [head]
        relation_ids = [relation]
        tail_ids = [tail]

        # weight
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
    def collate_fn_rr(data):
        inputs = torch.cat([_[0] for _ in data], dim=0)
        segment_ids = torch.cat([_[1] for _ in data], dim=0)
        attn_masks = torch.cat([_[2] for _ in data], dim=0)
        labels = torch.cat([_[3] for _ in data], dim=0)

        _mask = (labels == 1)
        in_ids1 = inputs[_mask]
        in_ids2 = inputs[~_mask]
        seg1 = segment_ids[_mask]
        seg2 = segment_ids[~_mask]
        attn1 = attn_masks[_mask]
        attn2 = attn_masks[~_mask]

        label_ids = labels.new_ones()  # labels should be 1 (margin ranking loss)
        return in_ids1, attn1, seg1, in_ids2, attn2, seg2, label_ids


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

