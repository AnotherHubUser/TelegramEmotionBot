import torch

class LearnableLayerPooling(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layer_weights = torch.nn.Parameter(torch.zeros(num_layers))

    def forward(self, x):
        # stacked_states = torch.stack(x.hidden_states)
        stacked_states = x # [B, 13, T_feats, 768]
        weights = torch.nn.functional.softmax(self.layer_weights, dim=0)
        # weighted_states_sum = torch.sum(stacked_states * weights.view(self.num_layers, 1, 1, 1), dim=0)
        weighted_states_sum = torch.sum(stacked_states * weights.view(1, self.num_layers, 1, 1), dim=1)
        return weighted_states_sum
