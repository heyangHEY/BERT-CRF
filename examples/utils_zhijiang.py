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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import pandas as pd
from io import open
import json
import argparse

import collections
from collections import OrderedDict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report
from tqdm import tqdm

logger = logging.getLogger(__name__)

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

    def __init__(self, input_ids, input_mask, segment_ids, unique_id=None, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.unique_id = unique_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class ZhiJiangProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.csv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train/train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev/dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/test.csv")), "test")

    def get_labels(self):
        """See base class."""
        label_list = []
        bi_list = ['A', 'O']
        cate_list = ['baozhuang', 'chengfen', 'chicun', 'fuwu', 'gongxiao', 'jiage', 'qiwei',
                     'shiyongtiyan', 'wuliu', 'xinxiandu', 'zhenwei', 'zhengti', 'qita']
        opinion_list = ['pos', 'neu', 'neg']
        
        label_list.append('[PAD]')
        label_list.append('O')
        label_list.append('[CLS]')
        label_list.append('[SEP]')
        for i in bi_list:
            for j in cate_list:
                bi_cate = i+'-'+j
                for k in opinion_list:
                    bi_cate_pola = bi_cate+'-'+k
                    label_list.append(bi_cate_pola) # 共2*3*13 + 4 = 82

        # id_to_label = dict(zip(label_list.values(), label_list.keys()))
        # id_to_label = {v: k for k, v in label_list.items()}
        file = open('label_list.json', 'w', encoding='utf-8')
        json.dump(label_list, file, ensure_ascii=False)
        file.close()

        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            if set_type != 'test':
                label = line[2]
                examples.append(InputExample(guid=guid, text_a=text_a, label=label))
            else:
                examples.append(InputExample(guid=guid, text_a=text_a))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0, # cls_token_segment_id是0不是1？
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index == 50:
            #import pdb;pdb.set_trace()
            pass
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        #tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = [char.lower() for char in example.text_a]
        labels = example.label.split()

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                labels = labels[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        labels_ = labels + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            labels_ = [cls_token]+labels_
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if output_mode == "classification":
            label_ids = [label_map[label] for label in labels_]
        else:
            raise KeyError(output_mode)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            label_ids = label_ids + ([pad_token] * padding_length)
            if len(label_ids)  == 129:
                import pdb;pdb.set_trace()

        unique_id = int(example.guid.split('-')[1])  # hey

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("unique_id: %s" % " ".join([str(unique_id)]))  # hey
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("labels: %s" % " ".join([str(x) for x in labels_]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              unique_id=unique_id)) # hey
    return features


def convert_test_examples_to_features(examples, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0, # cls_token_segment_id是0？
                                 mask_padding_with_zero=True):

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index == 50:
            # import pdb;pdb.set_trace()
            pass
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = [char.lower() for char in example.text_a]

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        unique_id = int(example.guid.split('-')[1]) # hey

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid)) # guid = "%s-%s" % (set_type, i)
            logger.info("unique_id: %s" % " ".join([str(unique_id)])) # hey
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          unique_id=unique_id)) # hey 已经将str的guid转化为int类型
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    labels_list = labels.reshape(-1).tolist()
    preds_list = preds.reshape(-1).tolist()
    label_num = 82
    labels_ = [i for i in range(label_num)][4:]
    report = classification_report(labels_list, preds_list, labels=labels_)
    return {
        "report": report
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "zhijiang":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

'''
四、评分标准
1、相同ID内逐一匹配各四元组，若AspectTerm，OpinionTerm，Category，Polarity四个字段均正确，则该四元组正确；
2、预测的四元组总个数记为P；真实标注的四元组总个数记为G；正确的四元组个数记为S：
（1）精确率：Precision=S/P
（2）召回率：Recall=S/G
（3）F值:F1-score=\frac{2*Precision*Recall}{Precision+Recall}
我们以 F1-score 作为最终的评测指标进行排名。
{'guid', 'AspectTerms', 'OpinionTerms', 'Categories', 'Polarities'}
'''
def compute_quad_metrics(pred_file, label_file):
    labels = pd.read_csv(label_file, sep=',')
    preds = pd.read_csv(pred_file, sep=',')

    G = labels.shape[0]
    P = preds.shape[0]
    s_list = []

    label_ids = labels['guid'].unique().tolist()
    for id_ in label_ids:
        label_data = labels[labels['guid'] == id_]
        pred_data = preds[preds['guid'] == id_]
        for i in range(pred_data.shape[0]): # hey 此处逻辑上有漏洞。默认pred中同个guid的四元组不重复，label中也是
            pred_sample = pred_data.iloc[i]
            for j in range(label_data.shape[0]):
                s = 0.0
                label_sample = label_data.iloc[j]
                if pred_sample['OpinionTerms'] == label_sample['OpinionTerms'] and \
                        pred_sample['AspectTerms'] == label_sample['AspectTerms'] and \
                        pred_sample['Polarities'] == label_sample['Polarities'] and \
                        pred_sample['Categories'] == label_sample['Categories']:
                    s = 1
                s_list.append(s)
    precision = sum(s_list) / P
    recall = sum(s_list) / G
    if precision + recall != 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.
    print('precision:{:.2f},recall:{:.2f},f1_score:{:.2f}'.format(precision*100, recall*100, f1_score*100))
    return {"precision": precision, "recall": recall, "f1_score": f1_score}


processors = {
    "zhijiang": ZhiJiangProcessor,
}

output_modes = {
    "zhijiang": "classification",
}

def get_metrics(data_dir='/ZJL/data/', gold_result_file='test/gold_Result.csv', pred_result_file='test/pred_Result_.csv'):
    gold_result = pd.read_csv(data_dir+gold_result_file, sep='\t')
    pred_result = pd.read_csv(data_dir+pred_result_file, sep='\t')
    
    G = gold_result.shape[0]
    P = pred_result.shape[0]
    s_list = []
    
    gold_ids = gold_result['ID'].unique().tolist()
    for id_ in gold_ids:
        gold_data = gold_result[gold_result['ID'] == id_]
        pred_data = pred_result[pred_result['ID'] == id_]
        for i in range(pred_data.shape[0]):
            pred_sample = pred_data.iloc[i]
            for j in range(gold_data.shape[0]):
                s1 = 0.0
                s2 = 0.0
                s = 0.0
                gold_sample = gold_data.iloc[j]
                if pred_sample['OpinionTerms'] == gold_sample['OpinionTerms']:
                    if pred_sample['AspectTerms'] == gold_sample['AspectTerms']:
                        s1 = 1        
                    if pred_sample['Polarities'] == gold_sample['Polarities'] and pred_sample['Categories'] == gold_sample['Categories']:
                        s2 = 1        
                    s = s1 * 0.6 + s2 * 0.4
                s_list.append(s)
    precision = sum(s_list) / P
    recall = sum(s_list) / G
    f1_score = 2 * precision * recall / (precision+recall)
    print('precision:{},recall:{},f1_score:{}'.format(precision, recall, f1_score))


def split_train_and_dev(data_dir, train_file, dev_file, split_ratio):
    from sklearn.model_selection import train_test_split
    train_raw = pd.read_csv(data_dir + train_file, sep='\t')

    train_file = open(data_dir + train_file, 'w', encoding='utf-8')
    dev_file = open(data_dir + dev_file, 'w', encoding='utf-8')

    train_file.write('ID\tReviews\tLabels\n')
    dev_file.write('ID\tReviews\tLabels\n')

    x, y = train_raw.iloc[:, :-1], train_raw['Labels']
    train_x, dev_x, train_y, dev_y = train_test_split(x, y, test_size=split_ratio,
                                                      random_state=0)
    for a, b, c in zip(train_x['ID'], train_x['Reviews'], train_y):
        train_file.write(str(a) + '\t' + b + '\t' + c + '\n')
    train_file.close()
    for a, b, c in zip(dev_x['ID'], dev_x['Reviews'], dev_y):
        dev_file.write(str(a) + '\t' + b + '\t' + c + '\n')
    dev_file.close()

# def kfold_train_and_dev(data_dir, train_file, dev_file, n_splits=5):
#     from sklearn.model_selection import KFold, StratifiedKFold
#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
#
#     train_raw = pd.read_csv(data_dir + train_file, sep='\t')
#     dev_file = open(data_dir + dev_file, 'w', encoding='utf-8')
#     train_file = open(data_dir + train_file, 'w', encoding='utf-8')
#     x, y = train_raw.iloc[:, :-1], train_raw['Labels']
#     splits = list(kf.split(x, y))
#
#     for a, b, c in zip(train_x['ID'], train_x['Reviews'], train_y):
#         train_file.write(str(a) + '\t' + b + '\t' + c + '\n')
#     train_file.close()
#     for a, b, c in zip(dev_x['ID'], dev_x['Reviews'], dev_y):
#         dev_file.write(str(a) + '\t' + b + '\t' + c + '\n')
#     dev_file.close()


def pretreatment_input_file(data_dir, review_file, pretreatment_file, result_file=None):
    review = pd.read_csv(data_dir + review_file, sep=',')
    writer = open(data_dir + pretreatment_file, 'w', encoding='utf-8')
    if result_file is not None:
        result = pd.read_csv(data_dir+result_file, sep=',')
        writer.write('ID\tReviews\tLabels\n')
        polar_dict = {'正面': 'pos',
                      '负面': 'neg',
                      '中性': 'neu'}
        cate_dict = {'包装': 'baozhuang',
                     '成分': 'chengfen',
                     '尺寸': 'chicun',
                     '服务': 'fuwu',
                     '功效': 'gongxiao',
                     '价格': 'jiage',
                     '气味': 'qiwei',
                     '使用体验': 'shiyongtiyan',
                     '物流': 'wuliu',
                     '新鲜度': 'xinxiandu',
                     '真伪': 'zhenwei',
                     '整体': 'zhengti',
                     '其他': 'qita'}
    else:
        writer.write('ID\tReviews\n')

    ids = OrderedDict(review['id']).values()
    for id_ in tqdm(ids):
        id_review = review[review['id'] == id_]
        review_ = id_review.iloc[0]['Reviews']
        if result_file is not None:
            id_result = result[result['id'] == id_]
            label = ['O'] * len(review_)
            for i in range(id_result.shape[0]):
                per_result = id_result.iloc[i]
                aspect_terms = per_result['AspectTerms']
                opinion_terms = per_result['OpinionTerms']
                cate = cate_dict[ per_result['Categories']]
                polar = polar_dict[ per_result['Polarities']]
                if aspect_terms != '_':
                    aspect_begin = int(per_result['A_start'])
                    aspect_end = int(per_result['A_end'])
                    for j in range(aspect_begin, aspect_end):
                        # if j == aspect_begin:
                        #     label[j] = 'B-'+cate+'-'+polar
                        # else:
                        #     label[j] = 'I-'+cate+'-'+polar
                        label[j] = 'A-' + cate + '-' + polar
                if opinion_terms != '_':
                    opinion_begin = int(per_result['O_start'])
                    opinion_end = int(per_result['O_end'])
                    for k in range(opinion_begin, opinion_end):
                        # if k == opinion_begin:
                        #     label[k] = 'B-'+cate+'-'+polar
                        # else:
                        #     label[k] = 'I-'+cate+'-'+polar
                        label[k] = 'O-' + cate + '-' + polar
            writer.write(str(id_)+'\t'+review_+'\t'+' '.join(label)+'\n')
        else:
            writer.write(str(id_) + '\t' + review_ + '\n')
    writer.close()

#quate_format: AspectTerms,OpinionTerms,Categories,Polarities
def translate_to_quate_format(guid, text, pred, label_list): # task 1 : label->四元组，再对四元组做f1评分
    polar_dict = {'pos': '正面',
                  'neg': '负面',
                  'neu': '中性'}
    cate_dict = {'baozhuang': '包装',
                 'chengfen': '成分',
                 'chicun': '尺寸',
                 'fuwu': '服务',
                 'gongxiao': '功效',
                 'jiage': '价格',
                 'qiwei': '气味',
                 'shiyongtiyan': '使用体验',
                 'wuliu': '物流',
                 'xinxiandu': '新鲜度',
                 'zhenwei': '真伪',
                 'zhengti': '整体',
                 'qita': '其他'}
    frame = {'guid': guid, 'AspectTerms': [], 'OpinionTerms': [], 'Categories': None, 'Polarities': None}
    frames = []
    logits = [label_list[logit] for logit in pred]
    for char, logit in zip(text, logits):
        if char in ['，', '。']:
            if frame['Categories']:
                frame['AspectTerms'] = ''.join(frame['AspectTerms']) if frame['AspectTerms'] else '_'
                frame['OpinionTerms'] = ''.join(frame['OpinionTerms']) if frame['OpinionTerms'] else '_'
                frames.append(frame)
                frame = {'guid': guid, 'AspectTerms': [], 'OpinionTerms': [], 'Categories': None, 'Polarities': None}
        else:
            if logit == 'O' or logit == '[CLS]' or logit == '[SEP]' or logit == '[PAD]':
                continue
            bi, category, polarity = logit.split('-')
            frame['Polarities'] = polar_dict[polarity]
            frame['Categories'] = cate_dict[category]
            if bi == 'A':
                frame['AspectTerms'].append(char)
            if bi == 'O':
                frame['OpinionTerms'].append(char)
    if frame['Categories']:
        frame['AspectTerms'] = ''.join(frame['AspectTerms']) if frame['AspectTerms'] else '_'
        frame['OpinionTerms'] = ''.join(frame['OpinionTerms']) if frame['OpinionTerms'] else '_'
        frames.append(frame)
    if len(frames) == 0:
        frames = [{'guid': guid, 'AspectTerms': '_', 'OpinionTerms': '_', 'Categories': '_', 'Polarities': '_'}]
    return frames


'''
results = {'ID': [],
           'Reviews': [],
           'Labels': []}
write: id,AspectTerms,OpinionTerms,Categories,Polarities
'''
def write_results_to_csv(results, output_dir, output_file, header=False):
    file = open('label_list.json', 'r', encoding='utf-8')
    label_list = json.load(file)
    frames = []
    for i in range(len(results['ID'])):
        guid = results['ID'][i].split('-')[1]
        text = results['Reviews'][i]
        pred = results['Labels'][i][1:len(text)+1]    # 只取[CLS]和[SEP]之间的数
        # assert(pred.index(label_list.index('[CLS]')) == 0)
        # end = pred.index(label_list.index('[SEP]'))
        # pred = pred[1:end]
        # assert(len(pred) == len(text))
        frame = translate_to_quate_format(guid, text, pred, label_list)
        frames.extend(frame)
    df = pd.DataFrame(frames, columns=['guid', 'AspectTerms', 'OpinionTerms', 'Categories', 'Polarities'])
    # df = pd.DataFrame(frames, columns=['ID', 'Reviews', 'Labels'])
    df.to_csv(output_dir+output_file, index=False, header=header)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the 'train', 'dev' and 'test' folders.")
    parser.add_argument("--train_review", default=None, type=str, required=True,
                        help="The train review dir")
    parser.add_argument("--train_result", default=None, type=str, required=True,
                        help="The train result dir")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The output directory where the train dataset will be written.")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
                        help="The output directory where the dev dataset will be written.")
    parser.add_argument("--test_review", default=None, type=str, required=True,
                        help="The test review dir")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="The output directory where the test dataset will be written.")
    parser.add_argument("--split_ratio", default=0.25, type=float, required=True,
                        help="The split ratio between the train file and the dev file.")
    args = parser.parse_args()

    print('pretreatment train and dev file')
    pretreatment_input_file(data_dir=args.data_dir, review_file=args.train_review,
                            result_file=args.train_result, pretreatment_file=args.train_file)
    split_train_and_dev(data_dir=args.data_dir, train_file=args.train_file,
                        dev_file=args.dev_file, split_ratio=args.split_ratio)

    print('pretreatment test file')
    pretreatment_input_file(data_dir=args.data_dir, review_file=args.test_review,
                            pretreatment_file=args.test_file)



