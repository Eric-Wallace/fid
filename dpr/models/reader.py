#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The reader model code + its utilities (loss computation and input batch tensor generator)
"""

import collections
import logging
from typing import List
import random

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor as T
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from dpr.data.reader_data import ReaderSample, ReaderPassage
from dpr.utils.model_utils import init_weights

logger = logging.getLogger()

ReaderBatch = collections.namedtuple('ReaderBatch', ['input_ids', 'answers_token_ids'])


class Reader(nn.Module):

    def __init__(self, encoder: nn.Module, hidden_size):
        super(Reader, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)
        init_weights([self.qa_outputs, self.qa_classifier])

    def forward(self, input_ids: T, attention_mask: T, start_positions=None, end_positions=None, answer_mask=None):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        start_logits, end_logits, relevance_logits = self._forward(input_ids.view(N * M, L),
                                                                   attention_mask.view(N * M, L))
        if self.training:
            return compute_loss(start_positions, end_positions, answer_mask, start_logits, end_logits, relevance_logits,
                                N, M)

        return start_logits.view(N, M, L), end_logits.view(N, M, L), relevance_logits.view(N, M)

    def _forward(self, input_ids, attention_mask):
        # TODO: provide segment values
        sequence_output, _pooled_output, _hidden_states = self.encoder(input_ids, None, attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        return start_logits, end_logits, rank_logits


class T5Reader(nn.Module):
    def __init__(self, encoder: nn.Module):
        super(T5Reader, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids: T, attention_mask: T, decoder_attn_mask: T = None, answers_token_ids=None):
        if answers_token_ids is not None and answers_token_ids != []:
            answers_token_ids[answers_token_ids == 0] = -100 # -100 is the canceled loss one
            loss = self.encoder(input_ids=input_ids, attention_mask=attention_mask, labels=answers_token_ids, decoder_attention_mask=decoder_attn_mask, return_dict=True).loss
            return loss
        else:
            # TODO, beam search here?
            generated_ids = self.encoder.generate(input_ids, attention_mask=attention_mask, use_cache=True, max_length=20)
            return generated_ids


def compute_loss(start_positions, end_positions, answer_mask, start_logits, end_logits, relevance_logits, N, M):
    start_positions = start_positions.view(N * M, -1)
    end_positions = end_positions.view(N * M, -1)
    answer_mask = answer_mask.view(N * M, -1)

    start_logits = start_logits.view(N * M, -1)
    end_logits = end_logits.view(N * M, -1)
    relevance_logits = relevance_logits.view(N * M)

    answer_mask = answer_mask.type(torch.FloatTensor).cuda()

    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    loss_fct = CrossEntropyLoss(reduce=False, ignore_index=ignored_index)

    # compute switch loss
    relevance_logits = relevance_logits.view(N, M)
    switch_labels = torch.zeros(N, dtype=torch.long).cuda()
    switch_loss = torch.sum(loss_fct(relevance_logits, switch_labels))

    # compute span loss
    start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask)
                    for (_start_positions, _span_mask)
                    in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]

    end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask)
                  for (_end_positions, _span_mask)
                  in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
                  torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)

    loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]
    span_loss = _calc_mml(loss_tensor)
    return span_loss + switch_loss


def create_reader_input(pad_token_id: int,
                        samples: List[ReaderSample],
                        passages_per_question: int,
                        max_length: int,
                        max_n_answers: int,
                        is_train: bool,
                        shuffle: bool,
                        ) -> ReaderBatch:
    """
    Creates a reader batch instance out of a list of ReaderSample-s
    :param pad_token_id: id of the padding token
    :param samples: list of samples to create the batch for
    :param passages_per_question: amount of passages for every question in a batch
    :param max_length: max model input sequence length
    :param max_n_answers: max num of answers per single question
    :param is_train: if the samples are for a train set
    :param shuffle: should passages selection be randomized
    :return: ReaderBatch instance
    """
    input_ids = []
    start_positions = []
    end_positions = []
    answers_masks = []
    answers_token_ids = []
    empty_sequence = torch.Tensor().new_full((max_length,), pad_token_id, dtype=torch.long)

    for sample in samples:
        ctxs = sample.passages

        sample_tensors = _create_question_passages_tensors(ctxs,
                                                           passages_per_question,
                                                           empty_sequence,
                                                           max_n_answers,
                                                           pad_token_id,
                                                           is_train,
                                                           is_random=shuffle)

        assert sample_tensors is not None
        sample_input_ids = sample_tensors
        input_ids.append(sample_input_ids)
        if is_train:
            answer_ids = random.choice(ctxs[0].answers_token_ids) # choose a random answer
            answer_ids = torch.cat((answer_ids, torch.ones(1).long()), dim=0) # add eos
            answers_token_ids.append(answer_ids)
    input_ids = pad_sequence(input_ids, batch_first=True)
    
    if is_train:
        answers_token_ids = pad_sequence(answers_token_ids, batch_first=True)

    return ReaderBatch(input_ids, answers_token_ids)


def _calc_mml(loss_tensor):
    marginal_likelihood = torch.sum(torch.exp(
        - loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)
    return -torch.sum(torch.log(marginal_likelihood +
                                torch.ones(loss_tensor.size(0)).cuda() * (marginal_likelihood == 0).float()))


def _pad_to_len(seq: T, pad_id: int, max_len: int):
    s_len = seq.size(0)
    if s_len > max_len:
        return seq[0: max_len]
    return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)


def _get_answer_spans(idx, positives: List[ReaderPassage], max_len: int):
    positive_a_spans = positives[idx].answers_spans
    return [span for span in positive_a_spans if (span[0] < max_len and span[1] < max_len)]


def _get_positive_idx(positives: List[ReaderPassage], max_len: int, is_random: bool):
    # select just one positive
    positive_idx = np.random.choice(len(positives)) if is_random else 0

    if not _get_answer_spans(positive_idx, positives, max_len):
        # question may be too long, find the first positive with at least one valid span
        positive_idx = next((i for i in range(len(positives)) if _get_answer_spans(i, positives, max_len)),
                            None)
    return positive_idx


def _create_question_passages_tensors(passages: List[ReaderPassage], total_size: int,
                                      empty_ids: T,
                                      max_n_answers: int,
                                      pad_token_id: int,
                                      is_train: bool,
                                      is_random: bool = True):
    max_len = empty_ids.size(0)
    negative_idxs = range(total_size)

    negatives_selected = [passages[i].sequence_ids for i in negative_idxs]
    negatives_selected = [item[0:max_len] for item in negatives_selected] # truncate long passages

    while len(negatives_selected) < total_size:
        negatives_selected.append(empty_ids.clone())

    input_ids = torch.cat(negatives_selected, dim=0)
    return input_ids
