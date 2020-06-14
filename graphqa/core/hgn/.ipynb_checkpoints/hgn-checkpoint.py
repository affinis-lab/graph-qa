import random
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize
import numpy as np

from transformers import BertModel, AutoTokenizer
from transformers import BertPreTrainedModel
from transformers import AdamW

from torch_geometric.nn import GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm_

from .utils import get_best_indexes
from .validation import validate
from ..optimizers import warmup_linear


class HierarchicalGraphNetwork(BertPreTrainedModel):

    def __init__(self, config, *args, **kwargs):
        super(HierarchicalGraphNetwork, self).__init__(config)

        self.bert = BertModel(config)

        self.use_gat = kwargs.get('use_gat', False)
        self.use_lstm = kwargs.get('use_lstm', False)

        self.dropout_prob = 0.2
        self.graph_embedding_size = 768
        self.n_attention_heads = 1

        self.n_paragraph_nodes = kwargs.get('n_paragraph_nodes', 40)
        self.n_sentence_nodes = kwargs.get('n_sentence_nodes', 4)
        self.n_graph_nodes = self.n_paragraph_nodes + self.n_sentence_nodes + 1

        # LSTM for paragraph, sentence and question spans
        if self.use_lstm:
            self.bi_lstm = nn.LSTM(input_size=config.hidden_size,
                                   hidden_size=config.hidden_size,
                                   num_layers=1,
                                   dropout=self.dropout_prob,
                                   bidirectional=True)

        self.cls_linear = nn.Linear(config.hidden_size, self.graph_embedding_size)

        # 2 is for the bidirectional lstm
        self.num_directions = 2 if self.use_lstm else 1

        # paragraph, sentence and question mlp's
        self.paragraph_embeddings_mlp = nn.Linear(self.num_directions*config.hidden_size, self.graph_embedding_size)
        self.sentence_embeddings_mlp = nn.Linear(self.num_directions*config.hidden_size, self.graph_embedding_size)
        self.question_embeddings_mlp = nn.Linear(self.num_directions*config.hidden_size, self.graph_embedding_size)

        # GAT convolutional layers
        if self.use_gat:
            self.conv1 = GATConv(
                self.graph_embedding_size,
                self.graph_embedding_size,
                heads=self.n_attention_heads,
                concat=False)

        # # Gated Attention part
        # dim = self.num_directions*config.hidden_size
        # self.w_m = Parameter(torch.FloatTensor(dim, dim).uniform_(-0.1, 0.1))  # 768x768
        # self.w_m_ = Parameter(torch.FloatTensor(dim, dim).uniform_(-0.1, 0.1))  # 768x768
        # self.w_s = Parameter(torch.FloatTensor(2*dim, 2*dim).uniform_(-0.1, 0.1))  # 1536x1536

        self.output_paragraph_linear = nn.Linear(1 * self.graph_embedding_size, 1)
        self.output_sentence_linear = nn.Linear(1 * self.graph_embedding_size, 1)

        qa_output_size = 2 * self.graph_embedding_size if self.use_gat else self.graph_embedding_size
        self.qa_outputs = nn.Linear(qa_output_size, 2)

        self.init_weights()

    def _filter_redundant_offsets(self, offsets, dim=0):
        valid_cols = []
        for col_idx in range(offsets.size(dim)):
            if not torch.all(offsets[col_idx] == 0):
                valid_cols.append(col_idx)
        return offsets[valid_cols]

    def _create_node_features(self, node_offsets, pooled_output):
        graph_embeddings = []
        for b_num, batch in enumerate(node_offsets):
            for start_offset, end_offset in batch:
                start_offset, end_offset = start_offset.item(), end_offset.item()
                if end_offset == 0:
                    graph_embeddings.append(self.num_directions*torch.zeros(pooled_output.size(-1)).to(self.device))
                else:
                    # TODO: dont forget to retain batch_size
                    if self.use_lstm:
                        span = pooled_output[b_num, start_offset:end_offset].unsqueeze(1)
                        lstm_out, (h_n, c_n) = self.bi_lstm(span)

                        lstm_out = lstm_out.squeeze(1)
                        graph_embeddings.append(lstm_out[-1])
                    else:
                        graph_embeddings.append(pooled_output[b_num, start_offset:end_offset].mean(dim=0))

        return torch.stack(graph_embeddings)

    def _apply_gated_attention(self, encoded_paragraphs, graph_embeddings, batch_size):
        param_expansion_size = (batch_size, self.w_m.size(0), self.w_m.size(1))  # (batch_size, hidden_size, hidden_size)

        C = torch.bmm(encoded_paragraphs, self.w_m.expand(param_expansion_size))
        H = torch.bmm(graph_embeddings.unsqueeze(0), self.w_m_.expand(param_expansion_size)).transpose(1, 2)

        C = torch.bmm(C, H)
        H_ = torch.bmm(C, graph_embeddings.unsqueeze(0))

        w_s_expansion_size = (batch_size, self.w_s.size(0), self.w_s.size(1))  # (batch_size, 2*hidden_size, 2*hidden_size)
        encoded_paragraphs = torch.bmm(torch.cat((encoded_paragraphs, H_), dim=2), self.w_s.expand(w_s_expansion_size))
        encoded_paragraphs = torch.sigmoid(encoded_paragraphs)

        return encoded_paragraphs

    def _get_qa_predictions(self, sequence_output):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

    def forward(self,
                input_ids,
                attention_mask,
                segment_ids,
                node_offsets,
                paragraph_node_indexes,
                sentence_node_indexes,
                edge_indexes=None):

        batch_size = input_ids.size(0)

        paragraph_node_indexes = paragraph_node_indexes[paragraph_node_indexes != -1]
        sentence_node_indexes = sentence_node_indexes[sentence_node_indexes != -1]

        encoded_paragraphs, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        pooled_output = encoded_paragraphs[:, 1:]

        graph_embeddings = self._create_node_features(node_offsets, pooled_output)

        graph_embeddings[paragraph_node_indexes] = self.paragraph_embeddings_mlp(graph_embeddings[paragraph_node_indexes].clone())
        graph_embeddings[sentence_node_indexes] = self.sentence_embeddings_mlp(graph_embeddings[sentence_node_indexes].clone())
        graph_embeddings[0] = self.question_embeddings_mlp(graph_embeddings[0].clone())

        if self.use_gat:
            edge_indexes = edge_indexes.squeeze(0)
            edge_indexes = self._filter_redundant_offsets(edge_indexes)
            edge_indexes = edge_indexes.t().contiguous()

            # TODO: retain batch_size
            graph_embeddings = self.conv1(graph_embeddings, edge_indexes)

            encoded_paragraphs = self._apply_gated_attention(encoded_paragraphs, graph_embeddings, batch_size)

        paragraph_embeddings = graph_embeddings[paragraph_node_indexes]
        sentence_embeddings = graph_embeddings[sentence_node_indexes]

        paragraph_predictions = self.output_paragraph_linear(paragraph_embeddings)
        sentence_predictions = self.output_sentence_linear(sentence_embeddings)

        start_logits, end_logits = self._get_qa_predictions(encoded_paragraphs)

        return paragraph_predictions, sentence_predictions, start_logits, end_logits


class HGN:
    def __init__(
                self,
                pretrained,
                model_path=None,
                use_gat=False,
                use_lstm=False,
                max_paragraph_num=40,
                max_sentence_num=4,
                max_seq_len=512,
                random_seed=42):

        self.model_path = model_path

        self.max_paragraph_num = max_paragraph_num
        self.max_sentence_num = max_sentence_num

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: enable loading trained models
        self.model = HierarchicalGraphNetwork.from_pretrained(pretrained, use_gat=use_gat, use_lstm=use_lstm)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

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

#         self.optimizer = RAdam(optimizer_grouped_parameters,
#                             lr=learning_rate)

    def fit(
            self,
            train_dataloader,
            train_batch_size=2,
            validation_dataloader=None,
            validation_batch_size=1,
            validate_after_steps=None,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            learning_rate=3e-5,
            warmup_proportion=0.1,
            verbose=True):

        # if not specified, validate after each epoch
        if not validate_after_steps:
            validate_after_steps = len(train_dataloader)

        if gradient_accumulation_steps > 1 and validate_after_steps % gradient_accumulation_steps != 0:
            # TODO: just display a warning in the future
            raise Exception('Modulo of validate_after_steps and gradient_accumulation_steps must be 0, if gradient_accumulation_steps is > 1.')

        num_train_steps = int(len(train_dataloader) // train_batch_size // gradient_accumulation_steps * num_train_epochs)

        self._prepare_optimizer(learning_rate)

        if self.model_path:
            pretrained_model = torch.load(self.model_path)
            self.model.load_state_dict(pretrained_model['model_state'])
            self.optimizer.load_state_dict(pretrained_model['optimizer_state'])

        global_step = 0
        global_optimizer_step = 0

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(num_train_epochs):
            training_loss = 0
            epoch_paragraph_loss = 0
            epoch_sentence_loss = 0
            epoch_qa_loss = 0

            progress_bar = tqdm(train_dataloader, desc='Training', position=0, leave=True)
            for step, batch in enumerate(progress_bar):
                batch = tuple(t.to(self.device) for t in batch)

                # TODO: spread to multiple lines for god's sake
                input_ids, input_masks, segment_ids, labels, edge_indexes, node_offsets, paragraph_node_indexes, sentence_node_indexes, start_positions, end_positions  = batch

                paragraph_preds, sentence_preds, start_logits, end_logits = self.model(
                            input_ids,
                            input_masks,
                            segment_ids,
                            node_offsets,
                            paragraph_node_indexes,
                            sentence_node_indexes,
                            edge_indexes=edge_indexes)

                # TODO: retain batch dim
                labels = labels.view(labels.size(1))

                start_positions, end_positions = start_positions.unsqueeze(0), end_positions.unsqueeze(0)
                start_positions = start_positions[start_positions != -1][0]
                end_positions = end_positions[end_positions != -1][0]
                start_positions, end_positions = start_positions.unsqueeze(0), end_positions.unsqueeze(0)

                paragraph_labels = labels[paragraph_node_indexes[paragraph_node_indexes != -1]].unsqueeze(1)
                sentence_labels = labels[sentence_node_indexes[sentence_node_indexes != -1]].unsqueeze(1)

                # maybe apply output mask to redundant nodes instead of filtering
                paragraph_loss = F.binary_cross_entropy_with_logits(paragraph_preds, paragraph_labels, reduction='mean')
                sentence_loss = F.binary_cross_entropy_with_logits(sentence_preds, sentence_labels, reduction='mean')

                loss = paragraph_loss + sentence_loss

                if start_positions is not None and end_positions is not None:
                    # If we are on multi-GPU, split add a dimension
                    if len(start_positions.size()) > 1:
                        start_positions = start_positions.squeeze(-1)
                    if len(end_positions.size()) > 1:
                        end_positions = end_positions.squeeze(-1)

                    # TODO: remove once batch size is introduced
                    start_logits, end_logits = start_logits.squeeze(1), end_logits.squeeze(1)

                    ignored_index = start_logits.size(1)
                    start_positions.clamp_(0, ignored_index)
                    end_positions.clamp_(0, ignored_index)

                    loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)

                    qa_loss = start_loss + end_loss
                    loss += qa_loss

                if self.n_gpu > 1:
                    loss = loss.mean()

                loss = loss / gradient_accumulation_steps
                loss.backward()

                training_loss += loss.item()
                epoch_paragraph_loss += paragraph_loss.item() / gradient_accumulation_steps
                epoch_sentence_loss += sentence_loss.item() / gradient_accumulation_steps
                if start_positions and end_positions:
                    epoch_qa_loss += qa_loss.item() / gradient_accumulation_steps

                global_step += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    lr_this_step = learning_rate * warmup_linear(global_optimizer_step/num_train_steps, warmup_proportion)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_this_step

                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_optimizer_step += 1

                if global_step % validate_after_steps == 0 and validation_dataloader:
                    validate(self,
                             validation_dataloader,
                             self.device,
                             num_validation_steps=400,
                             gradient_accumulation_steps=gradient_accumulation_steps
                             )
                    self.model.train()

                if verbose:
                    progress_bar.set_description("Epoch %i Loss: %.4f P: %.4f S: %.4f QA: %.4f"
                                                 % (int(epoch+1),
                                                    float(training_loss/(step+1)),
                                                    float(epoch_paragraph_loss/(step+1)),
                                                    float(epoch_sentence_loss/(step+1)),
                                                    float(epoch_qa_loss/(step+1))
                                                    ))

                ###
                # TODO: parameterize - save the model on every 2000 steps
                ###
                if global_step % 2000 == 0:
                    torch.save({
                                'model_state': self.model.state_dict(),
                                'optimizer_state': self.optimizer.state_dict(),
                            }, 'model_state_dict_1000.pth')

        return self.model

    def load(self, path):
        model_state_dict = torch.load(path)['model_state']
        self.model.load_state_dict(model_state_dict)

    def predict(self, question, paragraphs):
        answers = []
        for batch in paragraphs:
            texts = map(lambda paragraph: paragraph['text'], batch)

            inputs = self._convert_to_features(question, texts)
            paragraph_predictions, sentence_predictions, start_logits, end_logits = self.model(*inputs)

            start_indexes = get_best_indexes(start_logits.flatten(), n_best_size=1)[0]
            end_indexes = get_best_indexes(end_logits.flatten(), n_best_size=1)[0]

            input_ids = inputs[0]
            answer = self.tokenizer.decode(input_ids[start_indexes:end_indexes+1])
            answers.append(answer)

        return answers

    def _convert_to_features(self, question, paragraphs, max_length=512):
        tokens = [self.tokenizer.cls_token]

        question_tokens = self.tokenizer.tokenize(question)
        tokens += question_tokens
        tokens.append(self.tokenizer.sep_token)

        node_offsets = []
        paragraph_node_indexes, sentence_node_indexes = [], []

        segment_start = len(tokens)

        idx = 1
        for paragraph in paragraphs:
            paragraph_node_indexes.append(idx)
            idx += 1

            paragraph_offset_start = len(tokens)
            sentences = sent_tokenize(paragraph)

            sentence_offsets = []
            for sentence in sentences:
                sentence_node_indexes.append(idx)
                idx += 1

                sentence_offset_start = len(tokens)
                sentence_tokens = self.tokenizer.tokenize(sentence)
                tokens += sentence_tokens
                sentence_offset_end = len(tokens)
                sentence_offsets.append((sentence_offset_start, sentence_offset_end))

            tokens.append(self.tokenizer.sep_token)
            paragraph_offset_end = len(tokens)

            node_offsets += [(paragraph_offset_start, paragraph_offset_end)] + sentence_offsets

        segment_end = len(tokens)

        segment_ids = torch.zeros(max_length)
        segment_ids[segment_start:segment_end] = torch.ones(segment_end - segment_start)

        node_offsets = torch.tensor(node_offsets)

        encoding = self.tokenizer.encode_plus(
            tokens,
            add_special_tokens=False,
            max_length=max_length,
            pad_to_max_length=True,
            is_pretokenized=True,
            return_tensors='pt',
            return_token_type_ids=False,
        )

        input_ids = encoding['input_ids']
        input_mask = encoding['attention_mask']

        return {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'node_offsets': node_offsets,
            'paragraph_node_indexes': paragraph_node_indexes,
            'sentence_node_indexes': sentence_node_indexes,
        }
