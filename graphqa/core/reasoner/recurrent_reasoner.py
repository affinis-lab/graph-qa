from collections import defaultdict
import random

import numpy as np
from tqdm.auto import tqdm

from transformers import BertModel, BertTokenizer
from transformers import BertPreTrainedModel
from transformers import AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_

from .abstract_reasoner import AbstractReasoner
from .utils import batch_generator, shuffle_features, prepare_dataloader
from .metrics import intersection_ratio_metric, exact_match_metric
from ..optimizers import warmup_linear


class RecurrentReasonerModel(BertPreTrainedModel):

    def __init__(self, config, *args, **kwargs):
        super(RecurrentReasonerModel, self).__init__(config)

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initial state
        self.s = Parameter(torch.FloatTensor(config.hidden_size).uniform_(-0.1, 0.1))

        # Scaling factor for weight norm
        self.g = Parameter(torch.FloatTensor(1).fill_(1.0))

        # RNN weight
        self.rw = nn.Linear(2*config.hidden_size, config.hidden_size)

        # EOE and output bias
        self.eos = Parameter(torch.FloatTensor(config.hidden_size).uniform_(-0.1, 0.1))
        self.bias = Parameter(torch.FloatTensor(1).zero_())

        self.cpu = torch.device('cpu')

        self.init_weights()

    def weight_norm(self, state):
        state = state / state.norm(dim = 2).unsqueeze(2)
        state = self.g * state
        return state

    '''
    input_ids, token_type_ids, attention_mask: (B, N, L)
    B: batch size
    N: maximum number of Q-P pairs
    L: maximum number of input tokens
    '''
    def encode(self, input_ids, token_type_ids, attention_mask, split_chunk=None):
        B = input_ids.size(0)
        N = input_ids.size(1)
        L = input_ids.size(2)

        input_ids = input_ids.contiguous().view(B*N, L)
        token_type_ids = token_type_ids.contiguous().view(B*N, L)
        attention_mask = attention_mask.contiguous().view(B*N, L)

        if not split_chunk:
            # TODO: try concatenating last N layer embeddings - then hidden size of rw, s and eos will have to be multiplied by N as well
            encoded_layers, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) #, token_type_ids=token_type_ids)
            # [CLS] token embedding
            pooled_output = encoded_layers[:, 0]
            # alternative - averaged words embeddings (maybe skip the first [CLS] token and leave out padded words (mean on non-zero))
            # pooled_output = encoded_layers.mean(dim=1)
        else:
            idx = 0
            for chunk in batch_generator(list(zip(input_ids, token_type_ids, attention_mask)), split_chunk):
                input_ids_ = torch.stack(list(chunk[i][0] for i in range(len(chunk))))
                token_type_ids_ = torch.stack(list(chunk[i][1] for i in range(len(chunk))))
                attention_mask_ = torch.stack(list(chunk[i][2] for i in range(len(chunk))))

                encoded_layers, _ = self.bert(input_ids_, attention_mask_, token_type_ids_)
                encoded_layers = encoded_layers[:, 0]

                if idx == 0:
                    pooled_output = encoded_layers
                else:
                    pooled_output = torch.cat((pooled_output, encoded_layers), dim = 0)
                idx += 1

        paragraphs = pooled_output.view(pooled_output.size(0)//N, N, pooled_output.size(1))     # (B, N, D), D: BERT dim
        EOE = self.eos.unsqueeze(0).unsqueeze(0)    # (1, 1, D)
        EOE = EOE.expand(paragraphs.size(0), EOE.size(1), EOE.size(2))  # (B, 1, D)

        EOE = self.bert.encoder.layer[-1].output.LayerNorm(EOE)
        paragraphs = torch.cat((paragraphs, EOE), dim = 1) # (B, N+1, D)

        state = self.s.expand(paragraphs.size(0), 1, self.s.size(0))
        state = self.weight_norm(state)

        return paragraphs, state

    '''
    input_ids, token_type_ids, attention_mask: (B, N, L)
    - B: batch size
    - N: maximum number of Q-P pairs
    - L: maximum number of input tokens

    output_mask, target: (B, max_num_steps, N+1)
    '''
    def forward(self, input_ids, token_type_ids, attention_mask, output_mask, target_labels, golden_indices, max_num_steps, split_chunk=None):

        paragraphs, state = self.encode(input_ids, token_type_ids, attention_mask, split_chunk)

        h = state
        for i in golden_indices:
            input = paragraphs[:, i:i+1, :]     # (B, 1, D) 
            state = torch.cat((state, input), dim = 2)  # (B, 1, 2*D)
            state = self.rw(state)  # (B, 1, D)
            state = self.weight_norm(state)
            h = torch.cat((h, state), dim = 1)  # ...--> (B, max_num_steps, D)

        h = self.dropout(h)
        output = torch.bmm(h, paragraphs.transpose(1, 2))   # (B, max_num_steps, N+1)
        output = output + self.bias

        loss = F.binary_cross_entropy_with_logits(output, target_labels, weight=output_mask, reduction='mean')
        return loss


class RecurrentReasoner(AbstractReasoner):
    def __init__(
                self,
                pretrained,
                model_path=None,
                num_reasoning_steps = 2,
                max_paragraph_num = 10,
                max_seq_len = 192,
                split_chunk = None,
                random_seed = 42):

        self.num_reasoning_steps = num_reasoning_steps
        self.max_paragraph_num = max_paragraph_num + 1
        self.max_seq_len = max_seq_len

        self.split_chunk = split_chunk

        self.cpu = torch.device('cpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # TODO: enable loading trained models
        self.model = RecurrentReasonerModel.from_pretrained(pretrained)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)

        if model_path:
            # TODO: load optimizer aswell
            self.load(model_path)

        self.n_gpu = torch.cuda.device_count()

        if self.n_gpu > 1:
            # TODO: don't forget about the model.module problem in validation
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    def _prepare_optimizer(self, learning_rate):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                            lr=learning_rate, correct_bias=False)

    def fit(
            self,
            train_features,
            train_batch_size=1,
            validation_features=None,
            validation_batch_size=1,
            validate_after_steps=None,
            exclude_eoe=False,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            learning_rate = 3e-5,
            warmup_proportion = 0.1,
            do_shuffle_features=False,
            verbose=True):

        if gradient_accumulation_steps > 1 and validate_after_steps % gradient_accumulation_steps != 0:
            # TODO: just display a warning in the future
            raise Exception('Modulo of validate_after_steps and gradient_accumulation_steps must be 0, if gradient_accumulation_steps is > 1.')

        # if not specified, validate after each epoch
        if not validate_after_steps:
            validate_after_steps = len(train_features)

        num_train_steps = int(len(train_features) // train_batch_size // gradient_accumulation_steps * num_train_epochs)

        self._prepare_optimizer(learning_rate)

        global_step = 0
        global_optimizer_step = 0
        max_intersection_ratio = 0
        max_exact_match = 0

        self.model.train()
        self.optimizer.zero_grad()

        validation_dataloader = prepare_dataloader(validation_features)

        # TODO: parameterize num_epoch_iterations
        num_epoch_iterations = 10
        train_chunk = len(train_features)//num_epoch_iterations

        for epoch in range(num_train_epochs):
            random.shuffle(train_features)

            for iteration, train_features_iteration in enumerate(batch_generator(train_features, train_chunk)):
                training_loss = 0
                train_dataloader = prepare_dataloader(train_features_iteration, batch_size=train_batch_size, shuffle=True)
                progress_bar = tqdm(train_dataloader, desc='Training', position=0, leave=True)

                for step, batch in enumerate(progress_bar):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_masks, segment_ids, output_masks, target_labels, _ = batch

                    golden_indices = list(range(self.num_reasoning_steps))
                    if do_shuffle_features:
                        # TODO: wrap all of this in an object...
                        input_ids, segment_ids, input_masks, target_labels, output_masks, golden_indices = shuffle_features(input_ids, segment_ids, input_masks, target_labels, self.max_paragraph_num-1, golden_indices, output_masks=output_masks) 

                    loss = self.model(
                                input_ids,
                                segment_ids,
                                input_masks,
                                output_masks,
                                target_labels,
                                golden_indices,
                                self.num_reasoning_steps+1,
                                split_chunk=None)

                    if self.n_gpu > 1:
                        loss = loss.mean()

                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    training_loss += loss.item()

                    global_step += 1
                    if (step + 1) % gradient_accumulation_steps == 0:
                        lr_this_step = learning_rate * warmup_linear(global_optimizer_step/num_train_steps, warmup_proportion)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                        clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        global_optimizer_step += 1

                        # validate only when there are no accumulated gradients
                        if global_step % validate_after_steps  == 0 and validation_dataloader:
                            validation_metrics = self.validate(
                                                            validation_dataloader,
                                                            validation_batch_size,
                                                            exclude_eoe=exclude_eoe,
                                                            do_shuffle_features=do_shuffle_features)

                            if validation_metrics['intersection_ratio'] > max_intersection_ratio or validation_metrics['exact_match'] > max_exact_match:
                                max_intersection_ratio = validation_metrics['intersection_ratio']
                                max_exact_match = validation_metrics['exact_match']
                                torch.save({
                                            'model_state': self.model.state_dict(),
                                            'optimizer_state': self.optimizer.state_dict(),
                                        }, 'best_model_state_dict.pth')


                    if verbose:
                        if not validation_dataloader or not (validate_after_steps < global_step) :
                            progress_bar.set_description("Epoch %i-%i Step %i Loss: %.6f" 
                                                        % (int(epoch+1), int(iteration+1),  int(step+1),  float(training_loss/(step+1))))
                        else:
                            progress_bar.set_description("Epoch %i-%i Step %i Loss: %.6f IR %.2f EM %.2f" 
                                                        % (int(epoch+1), int(iteration+1), int(step+1), float(training_loss/(step+1)),
                                                            float(validation_metrics['intersection_ratio']),
                                                            float(validation_metrics['exact_match'])))

                    ###
                    # TODO: parameterize - save the model on every 2000 steps
                    ###
                    if global_step % 2000 == 0:
                        torch.save({
                                    'model_state': self.model.state_dict(),
                                    'optimizer_state': self.optimizer.state_dict(),
                                }, 'model_state_dict_1000.pth')

        return self.model

    def _get_output_predictions(self, paragraph_embeddings, state):
        output = torch.bmm(state, paragraph_embeddings.transpose(1, 2)) # (beam, 1, N+1)
        output = output + self.model.bias
        output = torch.sigmoid(output)

        return output.to(self.device)

    def _omit_previous_predictions(self, output, paragraph_index_history):
        output[:,:,paragraph_index_history] = 0.0
        return output

    def predict_greedy(self, paragraph_embeddings, state):
        paragraph_index_history = []
        inference_paragraph_embeddings = torch.FloatTensor(paragraph_embeddings.size()).copy_(paragraph_embeddings).to(self.device)

        # +1 for EOE
        for reasoning_step_num in range(self.num_reasoning_steps+1):
            output =  self._get_output_predictions(inference_paragraph_embeddings, state)

            if reasoning_step_num != 0:
                output = self._omit_previous_predictions(output, paragraph_index_history)

            score, paragraph_index = torch.max(output, dim=2)
            score, paragraph_index = score.item(), paragraph_index.item()
            paragraph_index_history.append((paragraph_index, score))

            state = torch.cat((state, inference_paragraph_embeddings[:, paragraph_index:paragraph_index+1, :]), dim=2)
            state = self.model.rw(state)
            state = self.model.weight_norm(state)

        return paragraph_index_history


    # TODO: For now works only with batch_size = 1
    def validate(
                self,
                valid_dataloader,
                batch_size=None,
                do_shuffle_features=False,
                exclude_eoe=False):

        validation_metrics = defaultdict(int)

        num_validation_steps = len(valid_dataloader)

        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader, desc='Validation', position=0, leave=True)
            for step, batch in enumerate(progress_bar):
                batch = tuple(t.to(self.device) for t in batch)

                input_ids, input_masks, segment_ids, _, target_labels, _ = batch # [batch_num:batch_num+1]

                golden_indices = list(range(self.num_reasoning_steps))
                if do_shuffle_features:
                    input_ids, segment_ids, input_masks, target_labels, golden_indices = shuffle_features(input_ids, segment_ids, input_masks, target_labels, self.max_paragraph_num-1, golden_indices)

                paragraph_embeddings, init_state = self.model.encode(input_ids, segment_ids, input_masks, None)

                pred_paragraph_indexes = self.predict_greedy(paragraph_embeddings, init_state)

                if exclude_eoe:
                    ###
                    ## leave out the last paragraph index (the EOE index)
                    ## leave out the last reasoning path (EOE) from target labels, but don't leave out the last paragraph
                    ## the third dimension will remain the same and still be self.max_paragraph_num
                    ## target_labels - last number of the last dimension will always be zero
                    ## pred_one_hot_vectors - last number of the last dimension can be a paragraph index, depending on what the model predicts
                    ##                      - a bad model will predict the EOE index for a paragraph (eg. [10,2]) for self.max_paragraph_num = 11
                    ###
                    pred_paragraph_indexes = pred_paragraph_indexes[:-1]
                    target_labels = target_labels[:, :-1]

                target_indexes = torch.max(target_labels.cpu(), dim=2)[1][0].numpy()

                intersection_ratio = intersection_ratio_metric(pred_paragraph_indexes, target_indexes)
                exact_match = exact_match_metric(pred_paragraph_indexes, target_indexes)       

                validation_metrics['intersection_ratio'] += intersection_ratio
                validation_metrics['exact_match'] += exact_match

                progress_bar.set_description("Step %i Intersection ratio %.2f Exact match %.2f" 
                                                % (int(step+1),
                                                float(validation_metrics['intersection_ratio']/(step+1)),
                                                float(validation_metrics['exact_match']/(step+1))))

            validation_metrics['intersection_ratio'] /= num_validation_steps
            validation_metrics['exact_match'] /= num_validation_steps

        self.model.train()

        return validation_metrics

    def load(self, path):
        model_state_dict = torch.load(path)
        self.model.load_state_dict(model_state_dict)

    def _tokenize_question(self, question):
        tokens_q = self.tokenizer.tokenize(question)
        tokens_q = ['[CLS]'] + tokens_q + ['[SEP]']
        return tokens_q

    def _tokenize_paragraph(self, paragraph, tokens_q):
        if len(tokens_q) > self.max_seq_len / 2:
            # ensure that the question doesn't take more than half of the input
            tokens_q = tokens_q[:int(self.max_seq_len / 2) - 1] + ['[SEP]']

        tokens_p = self.tokenizer.tokenize(paragraph)[:self.max_seq_len - len(tokens_q)-1]
        tokens_p = tokens_p + ['[SEP]']

        padding = [0] * (self.max_seq_len - len(tokens_p) - len(tokens_q))   

        input_ids_ = self.tokenizer.convert_tokens_to_ids(tokens_q + tokens_p)
        input_masks_ = [1] * len(input_ids_)
        segment_ids_ = [0] * len(tokens_q) + [1] * len(tokens_p)

        input_ids_ += padding
        input_masks_ += padding
        segment_ids_ += padding

        return input_ids_, input_masks_, segment_ids_

    def convert_to_features(self, question, paragraphs):
        paragraph_padding = [0] * self.max_seq_len

        tokens_q = self._tokenize_question(question)

        input_ids, input_masks, segment_ids = [], [], []
        for paragraph in paragraphs:
            if len(input_ids) == self.max_paragraph_num:
                break

            input_ids_, input_masks_, segment_ids_ = self._tokenize_paragraph(paragraph, tokens_q)
            input_ids.append(input_ids_)
            input_masks.append(input_masks_)
            segment_ids.append(segment_ids_)

        padding = [paragraph_padding] * (self.max_paragraph_num - len(input_ids))

        input_ids += padding
        input_masks += padding
        segment_ids += padding

        input_ids = torch.as_tensor(input_ids).unsqueeze(0).to(self.device)
        input_masks = torch.as_tensor(input_masks).unsqueeze(0).to(self.device)
        segment_ids = torch.as_tensor(segment_ids).unsqueeze(0).to(self.device)

        return input_ids, input_masks, segment_ids

    def rank(self, question, paragraphs):
        reranked_paragraphs = []
        for batch in paragraphs:
            texts = map(lambda paragraph: paragraph['text'], batch)
            input_ids, input_masks, segment_ids = self.convert_to_features(question, texts)

            paragraph_embeddings, init_state = self.model.encode(input_ids, segment_ids, input_masks)
            pred_paragraph_indexes = self.predict_greedy(paragraph_embeddings, init_state)

            selected_paragraphs = []
            scores = []
            for idx, score in pred_paragraph_indexes:
                if idx >= len(batch):
                    continue
                selected_paragraphs.append(batch[idx])
                scores.append(score)

            if selected_paragraphs:
                reranked_paragraphs.append({
                    'paragraphs': selected_paragraphs,
                    'score': sum(scores) / len(scores)
                })

        return reranked_paragraphs
