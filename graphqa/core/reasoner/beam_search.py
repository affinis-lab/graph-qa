import torch
import math
from torch.nn.parameter import Parameter


class BeamSearch:

    def __init__(self, config, batch_data, model, state):

        self.batch_data = batch_data
        self.batch_size = batch_data.size(0)

        self.beam_states = None
        self.current_paragraphs = None

        self.max_chain_len = config["max_chain_len"]
        self.max_paragraphs_num = config["max_paragraphs_num"]

        # Scaling factor for weight norm
        self.g = Parameter(torch.FloatTensor(1).fill_(1.0))

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.single_state = state
        self.beam_number = config["beam_number"]
        self.best_predictions = None
        self.ret_list = None

    def omit_previous_predictions(self):
        for ind, paragraph_index in enumerate(self.ret_list, start=0):
            self.best_predictions[ind, :, paragraph_index["data"]] = 0.0

    def run_beam_search(self):

        self.ret_list = [{"data": [], "score": 1} for x in range(self.beam_number)]

        self.current_paragraphs = torch.FloatTensor(self.max_paragraphs_num, self.batch_data.size(2)).zero_().to(self.device)
        self.current_paragraphs[:self.max_paragraphs_num, :].copy_(self.batch_data[0][:self.max_paragraphs_num, :]).to(self.device)
        self.develop_beam_chains()

        return sorted(self.ret_list, key=lambda beam_search: beam_search["score"], reverse=True)[0]

    def develop_beam_chains(self):

        self.beam_states = self.single_state.expand(self.beam_number, 1, self.single_state.size(2)).to(self.device)  # -> (beam, 1, D)

        for step_in_reasoning_chain in range(self.max_chain_len + 1):

            self.get_possible_beam_paragraphs(step_in_reasoning_chain)
            self.update_beam_states_with_new_predictions(step_in_reasoning_chain)


    def update_beam_states_with_new_predictions(self, step_in_reasoning_chain):

        score, paragraph_index = self.best_predictions.topk(k=self.beam_number, dim=2)
        list_of_paragraph_indexes = [p[0][0].item() for p in paragraph_index]
        score_list = [p[0][0].item() for p in score]

        if step_in_reasoning_chain == 0:
            list_of_paragraph_indexes = paragraph_index[0][0].cpu().numpy()
            score_list = score[0][0].cpu().detach().numpy()

        tensor_of_paragraph_indexes = torch.LongTensor(list_of_paragraph_indexes).to(self.device)
        input_paragraphs = self.current_paragraphs[tensor_of_paragraph_indexes].unsqueeze(1)
        for indx, beam_num_pred_index in enumerate(list_of_paragraph_indexes, start=0):
            self.ret_list[indx]["data"].append(beam_num_pred_index)
        for indx, beam_num_score_index in enumerate(score_list, start=0):
            self.ret_list[indx]["score"] *= math.log(beam_num_score_index / float(3 + 1e-6) * 0.7, 10)

        self.beam_states = torch.cat((self.beam_states, input_paragraphs), dim=2)
        with torch.no_grad():
            self.beam_states = self.model.rw(self.beam_states)
            self.beam_states = self.model.weight_norm(self.beam_states)


    def get_possible_beam_paragraphs(self, step_in_reasoning_chain):

        self.get_possible_predictions()
        if step_in_reasoning_chain > 0:
            self.omit_previous_predictions()

    def get_possible_predictions(self):

        output = torch.bmm(self.beam_states, self.current_paragraphs.unsqueeze(0).expand(
            self.beam_number, self.current_paragraphs.size(0), self.current_paragraphs.size(1)).transpose(1, 2))  # (beam, 1, N+1)
        output = output + self.model.bias
        output = torch.sigmoid(output)

        self.best_predictions = output.to(self.device)

