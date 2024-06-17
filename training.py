import torch
import torch.nn as nn
import torch.nn.functional as F


class BalRNN(nn.Module):
    '''one pop bal network with training on sqrt(K) weights'''

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 K=10,
                 J0I=0.4,
                 JII=-0.1):
        super(BalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.K = K
        self.J0I = J0I
        self.JII = JII
        #
        self.history = []
        # ff
        self.weight_ih = nn.ParameterList([
            nn.Parameter(
                torch.randn(hidden_size, input_size) *
                (J0I / torch.sqrt(torch.tensor(K, dtype=torch.float32))))
            for _ in range(num_layers)
        ])

        # Initialize hidden-to-hidden recurrent weights with masking
        self.weight_hh = nn.ParameterList([
            nn.Parameter(self.initialize_masked_weight(hidden_size, K, JII))
            for _ in range(num_layers)
        ])

    def initialize_ff_weight(self, input_size, hidden_size, K, J0I):
        indices = []
        values = []
        for i in range(hidden_size):
            selected_indices = torch.randperm(input_size)[:K]
            indices.extend([[i, j] for j in selected_indices])
            values.extend(
                [self.J0I / torch.sqrt(torch.tensor(K, dtype=torch.float32))] *
                K)  # list mult

        indices = torch.tensor(indices).t()
        values = torch.tensor(values, dtype=torch.float32)
        ff_sparse_weight = torch.sparse.FloatTensor(
            indices, values, torch.Size([hidden_size, input_size]))
        return ff_sparse_weight

    def initialize_sparse_weight(self, hidden_size, K, J):
        indices = []
        values = []
        for i in range(hidden_size):
            selected_indices = torch.randperm(hidden_size)[:int(
                K)]  # Choose sqrt(K) indices randomly
            indices.extend([[i, j] for j in selected_indices])
            values.extend(
                [J / torch.sqrt(torch.tensor(K, dtype=torch.float32))] *
                int(K))

        indices = torch.tensor(indices).t()
        values = torch.tensor(values, dtype=torch.float32)
        sparse_weight = torch.sparse.FloatTensor(
            indices, values, torch.Size([hidden_size, hidden_size]))
        return sparse_weight

    def initialize_masked_weight(self, hidden_size, K, J):
        mask_indices = []
        mask_values = []

        for i in range(hidden_size):
            # Randomly select K indices for connections
            selected_indices = torch.randperm(hidden_size)[:K]

            # Extend mask_indices with the selected indices for the current unit
            mask_indices.extend([[i, j] for j in selected_indices])

            # Set values for these connections to J/sqrt(K)
            mask_values.extend(
                [J / torch.sqrt(torch.tensor(K, dtype=torch.float32))] * K)

            # Randomly select sqrt(K) indices as trainable connections
            trainable_indices = selected_indices[:int(K**0.5)]
            trainable_values = [
                J / torch.sqrt(torch.tensor(int(K**0.5), dtype=torch.float32))
            ] * int(K**0.5)
            mask_values[-int(K**0.5):] = trainable_values

            # Create a trainable mask tensor to mark trainable weights
            trainable_mask = torch.zeros(hidden_size)
            trainable_mask[trainable_indices] = 1
            trainable_mask = trainable_mask.unsqueeze(1)
            trainable_mask = trainable_mask.repeat(1, hidden_size)
            trainable_mask = trainable_mask.reshape(-1)

        # Convert mask_indices and mask_values to tensors
        mask_indices = torch.tensor(mask_indices).t()
        mask_values = torch.tensor(mask_values, dtype=torch.float32)
        # Create the sparse tensor for the masked weight matrix
        masked_weight = torch.sparse.FloatTensor(
            mask_indices, mask_values, torch.Size([hidden_size, hidden_size]))

        # Create a boolean tensor to mark trainable weights and set requires_grad=True
        trainable_mask = torch.tensor(trainable_mask, dtype=torch.bool)
        masked_weight.requires_grad = True
        masked_weight.mask = trainable_mask

        return masked_weight

    def transfer_function(self, total_input):
        return F.relu(total_input)

    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Ensure (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h_t_minus_1 = h_0
        h_t = h_0.clone()
        output = []
        for t in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0:
                    h_t[layer] = self.transfer_function(
                        torch.sparse.mm(self.weight_ih[layer], x[t].T).T +
                        torch.sparse.mm(self.weight_hh[layer],
                                        h_t_minus_1[layer].T).T)
                else:
                    h_t[layer] = self.transfer_function(
                        torch.sparse.mm(self.weight_hh[layer], h_t[layer -
                                                                   1].T).T +
                        torch.sparse.mm(self.weight_hh[layer],
                                        h_t_minus_1[layer].T).T)
            output.append(h_t[-1])
            h_t_minus_1 = h_t.clone()
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_t


if __name__ == '__main__':
    input_size = 250  # num of ff neurons
    hidden_size = 1000  # num of neurons in the balanced network
    num_layers = 1  # 1 layer of balanced network
    batch_size = 1
    seq_len = 100  # simulation time
    K = 100
    J0I = 0.2
    JII = -0.1
    rnn = BalRNN(input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 K=K,
                 J0I=J0I,
                 JII=JII)
    # (batch_size, seq_len, input_size)
    x = 4.0 * torch.ones(batch_size, seq_len, input_size)
    # total_input will be: O(K * J0I / sqrt(K)) = O(sqrt(K))
    # randn(3, 5, input_size)
    h_0 = torch.zeros(num_layers, batch_size,
                      hidden_size)  # (num_layers, batch_size, hidden_size)
    output, h_n = rnn(x, h_0)
    print(output.shape)
    print(h_n.shape)

    import matplotlib.pyplot as plt
    import numpy as np

    # plt.hist(h_n.detach().numpy().squeeze(), 25)

    rates = output.detach().numpy().squeeze()
    #.mean(axis=1)
    # plt.plot(rates[:, :10])
    fig, ax = plt.subplots()
    ri = rates.mean(axis=0)
    print(ri.min(), ri.max(), sum(ri <= 0))
    ax.hist(ri, 25)
    ax.set_xscale('symlog')
    # ax.set_xticks(np.linspace(1e-5, 1e1, 1e-1))
    plt.title('rate distr: is this balanced??')
    plt.figure()
    plt.plot(rates.mean(axis=1))
    plt.title('network avg rate')
    print(sum(rates.mean(axis=1) <= 0))
    plt.show()
