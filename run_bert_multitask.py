# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import argparse
import logging
import random
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MarginRankingLoss
from sklearn import metrics
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformers import AdamW, BertTokenizer, BertConfig
from models import BertForSequenceClassification
from utils import *
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_list",
                        default=None,
                        type=str,
                        required=True,
                        help="The list of name of the tasks to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_debug",
                        action='store_true')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--eval_range', type=str, default='1:500')
    parser.add_argument('--eval_task', type=str, default="lp", choices=["lp", "rp"])
    parser.add_argument('--tb_log_dir', type=str, default="runs/null")
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--debug_index', type=int, default=0)
    args = parser.parse_args()

    summary = SummaryWriter(log_dir=args.tb_log_dir)
    task_list = args.task_list.split(",")

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    lp_processor = LPProcessor()
    rp_processor = RPProcessor()
    rr_processor = RRProcessor()

    lp_label_list = lp_processor.get_labels(args.data_dir)
    lp_num_labels = len(lp_label_list)
    rp_label_list = rp_processor.get_labels(args.data_dir)
    rp_num_labels = len(rp_label_list)
    rr_label_list = rr_processor.get_labels(args.data_dir)

    entity_list = lp_processor.get_entities(args.data_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    config = BertConfig.from_pretrained(args.bert_model)
    setattr(config, "lp_num_labels", lp_num_labels)
    setattr(config, "rp_num_labels", rp_num_labels)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, config=config)
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        task_total_dataset = dict()
        if "lp" in task_list:
            # load link prediction data
            train_bin_path = os.path.join(args.data_dir, 'train-lp.pt')
            if os.path.exists(train_bin_path):
                lp_train_data = torch.load(train_bin_path)
                logger.info("load %s" % train_bin_path)
            else:
                lp_train_examples = lp_processor.get_train_examples(args.data_dir)
                lp_train_features = lp_convert_examples_to_features(
                    lp_train_examples, lp_label_list, args.max_seq_length, tokenizer)
                all_input_ids = torch.tensor([f.input_ids for f in lp_train_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in lp_train_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in lp_train_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in lp_train_features], dtype=torch.long)
                lp_train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                torch.save(lp_train_data, train_bin_path)
            task_total_dataset["lp"] = lp_train_data
            logger.info("  [Link Prediction] Num examples = %d", len(lp_train_data))
        if "rp" in task_list:
            # load relation prediction data
            train_bin_path = os.path.join(args.data_dir, 'train-rp.pt')
            if os.path.exists(train_bin_path):
                rp_train_data = torch.load(train_bin_path)
                logger.info("load %s" % train_bin_path)
            else:
                rp_train_examples = rp_processor.get_train_examples(args.data_dir)
                rp_train_features = rp_convert_examples_to_features(
                    rp_train_examples, rp_label_list, args.max_seq_length, tokenizer)
                logger.info("  [Relation Prediction] Num examples = %d", len(rp_train_examples))
                all_input_ids = torch.tensor([f.input_ids for f in rp_train_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in rp_train_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in rp_train_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in rp_train_features], dtype=torch.long)
                rp_train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                torch.save(rp_train_data, train_bin_path)
            task_total_dataset["rp"] = rp_train_data
            logger.info("  [Relation Prediction] Num examples = %d", len(rp_train_data))
        if "rr" in task_list:
            # load margin rank data
            train_bin_path = os.path.join(args.data_dir, 'train-mr.pt')
            if os.path.exists(train_bin_path):
                mr_train_data = torch.load(train_bin_path)
                logger.info("load %s" % train_bin_path)
            else:
                mr_train_examples = rr_processor.get_train_examples(args.data_dir)
                train_features = rr_convert_examples_to_features(
                    mr_train_examples, rr_label_list, args.max_seq_length, tokenizer)
                logger.info("***** Running training *****")
                logger.info("  Num examples = %d", len(mr_train_examples))
                all_input_ids1 = torch.tensor([f.input_ids1 for f in train_features], dtype=torch.long)
                all_input_mask1 = torch.tensor([f.input_mask1 for f in train_features], dtype=torch.long)
                all_segment_ids1 = torch.tensor([f.segment_ids1 for f in train_features], dtype=torch.long)
                all_input_ids2 = torch.tensor([f.input_ids2 for f in train_features], dtype=torch.long)
                all_input_mask2 = torch.tensor([f.input_mask2 for f in train_features], dtype=torch.long)
                all_segment_ids2 = torch.tensor([f.segment_ids2 for f in train_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
                mr_train_data = TensorDataset(all_input_ids1, all_input_mask1, all_segment_ids1,
                                              all_input_ids2, all_input_mask2, all_segment_ids2, all_label_ids)
                torch.save(mr_train_data, train_bin_path)
            task_total_dataset["mr"] = mr_train_data
            logger.info("  [Margin Rank] Num examples = %d", len(mr_train_data))

        # get train loaders
        train_dataloader = {}
        for task in task_total_dataset:
            train_dataloader[task] = DataLoader(task_total_dataset[task],
                                                sampler=RandomSampler(task_total_dataset[task]),
                                                batch_size=args.train_batch_size)
        batch_nums = {task: len(train_dataloader[task]) for task in task_total_dataset}
        task_total_batch_num = sum([batch_nums[k] for k in batch_nums])

        # set batch order
        order = order_selection([task for task in task_total_dataset], batch_nums)

        # set recoders
        loss_recoder = {task: .0 for task in task_total_dataset}

        model.train()
        for k in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = {task: .0 for task in task_total_dataset}
            nb_tr_examples = {task: 0 for task in task_total_dataset}
            nb_tr_steps = 0

            task_iterators = {task: iter(train_dataloader[task]) for task in train_dataloader}
            for step, (cur_batch_task, index) in enumerate(tqdm(order, desc="Iteration over multi-tasks")):
                batch_some_task = next(task_iterators[cur_batch_task])

                if cur_batch_task in ["lp", "rp"]:
                    input_ids, input_mask, segment_ids, label_ids = tuple(t.to(device) for t in batch_some_task)
                    logits = model(input_ids, segment_ids, input_mask, labels=label_ids, task=cur_batch_task)
                    loss_fct = CrossEntropyLoss()
                    if cur_batch_task == "lp":
                        loss = loss_fct(logits.view(-1, lp_num_labels), label_ids.view(-1))
                        loss = loss * 1
                    else:  # task == "rp"
                        loss = loss_fct(logits.view(-1, rp_num_labels), label_ids.view(-1))
                        loss = loss * 1
                elif cur_batch_task in ["mr"]:
                    batch = tuple(t.to(device) for t in batch_some_task)
                    input_ids1, input_mask1, segment_ids1, input_ids2, input_mask2, segment_ids2, label_ids = batch
                    logits1 = model(input_ids1, segment_ids1, input_mask1, labels=None, task=cur_batch_task)
                    logits2 = model(input_ids2, segment_ids2, input_mask2, labels=None, task=cur_batch_task)
                    loss_fct = MarginRankingLoss(margin=args.margin)
                    loss = loss_fct(logits1, logits2, label_ids.view(-1))
                    loss = loss * 1
                else:
                    raise TypeError

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss[cur_batch_task] += loss.item()
                loss_recoder[cur_batch_task] += loss.item()
                nb_tr_examples[cur_batch_task] += label_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                summary.add_scalar('%s_training_loss' % cur_batch_task, loss.item(), global_step)
            # end of epoch
            # eval on dev set
            # eval_result = {}
            # eval_result["lp"] = dev_eval(args, device, global_step, lp_label_list, lp_num_labels, lp_processor, model,
            #                              tokenizer, lp_convert_examples_to_features, "lp")
            # eval_result["rp"] = dev_eval(args, device, global_step, rp_label_list, rp_num_labels, rp_processor, model,
            #                              tokenizer, rp_convert_examples_to_features, "rp")

            # for task in task_total_dataset:
            #     logger.info("[%s] Training loss: %.6f, Training examples: %d" % (task, tr_loss[task], nb_tr_examples[task]))
            #     for key in sorted(eval_result[task].keys()):
            #         logger.info("  %s = %s", key, str(eval_result[task][key]))
            # score = eval_result["lp"]["acc"]
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + "_%d" % (k+1))
            torch.save(model_to_save.state_dict(), output_model_file)
        # end of whole training

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.eval_task == "lp":
        # evaluate Link Prediction task
        train_triples = lp_processor.get_train_triples(args.data_dir)
        dev_triples = lp_processor.get_dev_triples(args.data_dir)
        test_triples = lp_processor.get_test_triples(args.data_dir)
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)

        eval_examples = lp_processor.get_test_examples(args.data_dir)
        eval_features = lp_convert_examples_to_features(
            eval_examples, lp_label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running Prediction *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None, task="lp")

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, lp_num_labels), label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        print(preds, preds.shape)
        
        all_label_ids = all_label_ids.numpy()

        preds = np.argmax(preds, axis=1)

        result = compute_metrics("kg", preds, all_label_ids)

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step

        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info("Triple classification acc is : %.4f" % metrics.accuracy_score(all_label_ids, preds))

        ############################################################################
        # run link prediction
        ############################################################################
        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        top_ten_hit_count = 0

        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])

        def _get_tensordataset(corrupt_list):
            tmp_examples = lp_processor._create_examples(corrupt_list, "test", args.data_dir)
            # print(len(tmp_examples))
            tmp_features = lp_convert_examples_to_features(tmp_examples, lp_label_list, args.max_seq_length, tokenizer,
                                                        print_info=False)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)
            return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        start, end = [int(x) for x in args.eval_range.split(":")]
        for idx, test_triple in enumerate(tqdm(test_triples[start-1:end]), start):
            head = test_triple[0]
            relation = test_triple[1]
            tail = test_triple[2]
            # print(test_triple, head, relation, tail)

            head_corrupt_list = [test_triple]
            tail_corrupt_list = [test_triple]
            # 4000 is magic number,
            # _entity_list = np.random.choice(entity_list, 4000, replace=False)
            for corrupt_ent in entity_list:
                if corrupt_ent != head:
                    tmp_triple = [corrupt_ent, relation, tail]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        head_corrupt_list.append(tmp_triple)
                if corrupt_ent != tail:
                    tmp_triple = [head, relation, corrupt_ent]
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str not in all_triples_str_set:
                        # may be slow
                        tail_corrupt_list.append(tmp_triple)
            head_corrupt = _get_tensordataset(head_corrupt_list)
            tail_corrupt = _get_tensordataset(tail_corrupt_list)

            eval_data = head_corrupt
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()
            preds = []
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                # label_ids = label_ids.to(device)
                
                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None, task="lp")
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)
                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0)       

            preds = preds[0]
            # get the dimension corresponding to current label 1
            #print(preds, preds.shape)
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            #print(rel_values, rel_values.shape)
            _, argsort1 = torch.sort(rel_values, descending=True)
            #print(max_values)
            #print(argsort1)
            argsort1 = argsort1.cpu().numpy()
            rank1 = np.where(argsort1 == 0)[0][0]

            ranks.append(rank1+1)
            ranks_left.append(rank1+1)
            if rank1 < 10:
                top_ten_hit_count += 1

            eval_data = tail_corrupt
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()
            preds = []
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                # label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None, task="lp")
                if len(preds) == 0:
                    batch_logits = logits.detach().cpu().numpy()
                    preds.append(batch_logits)
                else:
                    batch_logits = logits.detach().cpu().numpy()
                    preds[0] = np.append(preds[0], batch_logits, axis=0) 

            preds = preds[0]
            # get the dimension corresponding to current label 1
            rel_values = preds[:, all_label_ids[0]]
            rel_values = torch.tensor(rel_values)
            _, argsort1 = torch.sort(rel_values, descending=True)
            argsort1 = argsort1.cpu().numpy()
            rank2 = np.where(argsort1 == 0)[0][0]
            ranks.append(rank2+1)
            ranks_right.append(rank2+1)
            if rank2 < 10:
                top_ten_hit_count += 1

            logging.info('[%d/%d] left: %d, rgiht: %d, mean rank: %.4f, hit@10: %.4f' % (
                idx, len(test_triples), rank1, rank2, np.mean(ranks), (top_ten_hit_count * 1.0 / len(ranks))))

            file_prefix = str(args.data_dir[7:]) + "_" + str(args.train_batch_size) + "_" + str(args.learning_rate) + \
                          "_" + str(args.max_seq_length) + "_" + str(args.num_train_epochs) + "_" + str(args.eval_range)

            with open(os.path.join(args.output_dir, file_prefix + '_ranks.txt'), 'a') as f:
                f.write('\t'.join([str(idx), str(rank1), str(rank2)]) + '\n')
            # this could be done more elegantly, but here you go
            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)

        for i in [0,2,9]:
            logger.info('Hits left @{0}: {1}'.format(i+1, np.mean(hits_left[i])))
            logger.info('Hits right @{0}: {1}'.format(i+1, np.mean(hits_right[i])))
            logger.info('Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
        logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1./np.array(ranks_left))))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1./np.array(ranks_right))))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.eval_task == "rp":
        train_triples = rp_processor.get_train_triples(args.data_dir)
        dev_triples = rp_processor.get_dev_triples(args.data_dir)
        test_triples = rp_processor.get_test_triples(args.data_dir)
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)

        eval_examples = rp_processor.get_test_examples(args.data_dir)
        eval_features = rp_convert_examples_to_features(
            eval_examples, rp_label_list, args.max_seq_length, tokenizer)
        logger.info("***** [Relation Prediction] Running Prediction *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask, labels=None, task="rp")
                logits = outputs[0]  # if labels is None, outputs = (logits, hidden_states, attentions)

            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, rp_num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        print(preds, preds.shape)

        all_label_ids = all_label_ids.numpy()

        ranks = []
        filter_ranks = []
        hits = []
        hits_filter = []
        for i in range(10):
            hits.append([])
            hits_filter.append([])

        for i, pred in enumerate(preds):
            rel_values = torch.tensor(pred)
            _, argsort1 = torch.sort(rel_values, descending=True)
            argsort1 = argsort1.cpu().numpy()

            rank = np.where(argsort1 == all_label_ids[i])[0][0]
            # print(argsort1, all_label_ids[i], rank)
            ranks.append(rank + 1)
            test_triple = test_triples[i]
            filter_rank = rank
            for tmp_label_id in argsort1[:rank]:
                tmp_label = rp_label_list[tmp_label_id]
                tmp_triple = [test_triple[0], tmp_label, test_triple[2]]
                # print(tmp_triple)
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str in all_triples_str_set:
                    filter_rank -= 1
            filter_ranks.append(filter_rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

                if filter_rank <= hits_level:
                    hits_filter[hits_level].append(1.0)
                else:
                    hits_filter[hits_level].append(0.0)

        print("Raw mean rank: ", np.mean(ranks))
        print("Filtered mean rank: ", np.mean(filter_ranks))
        for i in [0, 2, 9]:
            print('Raw Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
            print('hits_filter Hits @{0}: {1}'.format(i + 1, np.mean(hits_filter[i])))
        preds = np.argmax(preds, axis=1)

        result = compute_metrics("kg", preds, all_label_ids)
        loss = tr_loss / nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "rp_test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        # relation prediction, raw
        print("Relation prediction hits@1, raw...")
        print(metrics.accuracy_score(all_label_ids, preds))

    if args.do_debug and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and args.eval_task == "lp":
        ############################################################################
        # DEBUGGING!!
        ############################################################################
        train_triples = lp_processor.get_train_triples(args.data_dir)
        dev_triples = lp_processor.get_dev_triples(args.data_dir)
        test_triples = lp_processor.get_test_triples(args.data_dir)
        all_triples = train_triples + dev_triples + test_triples

        all_triples_str_set = set()
        for triple in all_triples:
            triple_str = '\t'.join(triple)
            all_triples_str_set.add(triple_str)

        entity2text = lp_processor.get_entity2text(args.data_dir)

        ranks = []
        ranks_left = []
        ranks_right = []

        hits_left = []
        hits_right = []
        hits = []

        top_ten_hit_count = 0

        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])

        def _get_tensordataset(corrupt_list):
            tmp_examples = lp_processor._create_examples(corrupt_list, "test", args.data_dir)
            # print(len(tmp_examples))
            tmp_features = lp_convert_examples_to_features(tmp_examples, lp_label_list, args.max_seq_length, tokenizer,
                                                           print_info=False)
            all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in tmp_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in tmp_features], dtype=torch.long)
            # all_label_ids = torch.tensor([f.label_id for f in tmp_features], dtype=torch.long)
            return TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        test_triple = test_triples[args.debug_index]

        head = test_triple[0]
        relation = test_triple[1]
        tail = test_triple[2]
        # print(test_triple, head, relation, tail)

        head_corrupt_list = [test_triple]
        tail_corrupt_list = [test_triple]

        for corrupt_ent in entity_list:
            if corrupt_ent != head:
                tmp_triple = [corrupt_ent, relation, tail]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    head_corrupt_list.append(tmp_triple)
            if corrupt_ent != tail:
                tmp_triple = [head, relation, corrupt_ent]
                tmp_triple_str = '\t'.join(tmp_triple)
                if tmp_triple_str not in all_triples_str_set:
                    # may be slow
                    tail_corrupt_list.append(tmp_triple)
        head_corrupt = _get_tensordataset(head_corrupt_list)
        tail_corrupt = _get_tensordataset(tail_corrupt_list)

        model = BertForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        model.eval()

        eval_data = head_corrupt
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        preds = []
        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None, task="lp")
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)
            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)

        preds = preds[0]
        # get the dimension corresponding to current label 1
        # print(preds, preds.shape)
        # rel_values = preds[:, all_label_ids[0]]
        rel_values = preds[:, 1]
        rel_values = torch.tensor(rel_values)
        # print(rel_values, rel_values.shape)
        argvalues, argsort1 = torch.sort(rel_values, descending=True)
        # print(max_values)
        # print(argsort1)
        argvalues = argvalues.cpu().numpy()
        argsort1 = argsort1.cpu().numpy()
        rank1 = np.where(argsort1 == 0)[0][0]

        ranks.append(rank1 + 1)
        ranks_left.append(rank1 + 1)
        if rank1 < 10:
            top_ten_hit_count += 1

        # with open("debugging_result_left.txt", "w") as f:
        #     f.write(str(test_triple) + "\n")
        #     f.write("\t".join([entity2text[head], relation, entity2text[tail]]) + "\n")
        #     f.write("rank: %d\n\n" % rank1)
        #     for idx in range(len(argsort1)):
        #         __idx = argsort1[idx]
        #         h, _, _ = head_corrupt_list[__idx]
        #         f.write('\t'.join([str(idx), str(argvalues[idx]), entity2text[h]]) + '\n')
        print("====== head corrupt result =======")
        print(test_triple)
        print("<%s, %s, %s>" % (entity2text[head].split(",")[0], relation, entity2text[tail].split(",")[0]))
        print("rank: ", rank1+1)
        scores, words, entities = [], [], []
        for j in range(10):
            __idx = argsort1[j]
            h, _, _ = head_corrupt_list[__idx]
            scores.append(str(round(argvalues[j], 4)))
            words.append(entity2text[h].split(",")[0])
            entities.append(h)
        for item in [scores, words, entities]:
            for x in item:
                print(x)

        eval_data = tail_corrupt
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        preds = []
        for input_ids, input_mask, segment_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None, task="lp")
            if len(preds) == 0:
                batch_logits = logits.detach().cpu().numpy()
                preds.append(batch_logits)
            else:
                batch_logits = logits.detach().cpu().numpy()
                preds[0] = np.append(preds[0], batch_logits, axis=0)

        preds = preds[0]
        # get the dimension corresponding to current label 1
        rel_values = preds[:, 1]
        rel_values = torch.tensor(rel_values)
        argvalues, argsort1 = torch.sort(rel_values, descending=True)
        argvalues = argvalues.cpu().numpy()
        argsort1 = argsort1.cpu().numpy()
        rank2 = np.where(argsort1 == 0)[0][0]
        ranks.append(rank2 + 1)
        ranks_right.append(rank2 + 1)
        if rank2 < 10:
            top_ten_hit_count += 1

        logging.info('left: %d, rgiht: %d, mean rank: %.4f, hit@10: %.4f' % (
            rank1, rank2, np.mean(ranks), (top_ten_hit_count * 1.0 / len(ranks))))

        # with open("debugging_result_right.txt", "w") as f:
        #     f.write(str(test_triple) + "\n")
        #     f.write("\t".join([entity2text[head], relation, entity2text[tail]]) + "\n")
        #     f.write("rank: %d\n\n" % rank2)
        #     for idx in range(len(argsort1)):
        #         __idx = argsort1[idx]
        #         h, r, t = tail_corrupt_list[__idx]
        #         f.write('\t'.join([str(idx), str(argvalues[idx]), entity2text[t]]) + '\n')

        print("====== tail corrupt result =======")
        print(test_triple)
        print("<%s, %s, %s>" % (entity2text[head].split(",")[0], relation, entity2text[tail].split(",")[0]))
        print("rank: ", rank2+1)
        scores, words, entities = [], [], []
        for j in range(10):
            __idx = argsort1[j]
            _, _, t = tail_corrupt_list[__idx]
            scores.append(str(round(argvalues[j], 4)))
            words.append(entity2text[t].split(",")[0])
            entities.append(t)
        for item in [scores, words, entities]:
            for x in item:
                print(x)


def dev_eval(args, device, global_step, label_list, num_labels, processor, model, tokenizer,
             convert_examples_to_features, task):
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids, segment_ids, input_mask, labels=None, task=task)
            logits = outputs[0]

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    result = compute_metrics("kg", preds, all_label_ids.numpy())
    result['eval_loss'] = eval_loss
    result['global_step'] = global_step
    return result


if __name__ == "__main__":
    main()
