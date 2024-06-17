import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomRNN(nn.Module):
'''One pop balanced network
'''
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 K=10,
                 J0I=0.4,
                 JII=-0.1):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.J0I = J0I
        self.JII = JII
        #
        self.history = []
        #
        xx = 2.0 * torch.tensor(K) * torch.ones(batch_size, seq_len,
                                                input_size)

        self.weight_ih = nn.ParameterList([
            nn.Parameter(
                torch.randn(hidden_size, input_size) *
                (J0I / torch.sqrt(torch.tensor(K, dtype=torch.float32))))
            for _ in range(num_layers)
        ])

        # self.bias_ih = nn.ParameterList([
        #     nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)
        # ])

        self.weight_hh = nn.ParameterList([
            nn.Parameter(self.initialize_sparse_weight(hidden_size, K, JII))
            for _ in range(num_layers)
        ])

        # self.bias_hh = nn.ParameterList([
        #     nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)
        # ])
        # ww = nn.ParameterList([
        #     nn.Parameter(
        #         self.initialize_ff_weight(input_size, hidden_size, K, J0I))
        #     for _ in range(num_layers)
        # ])

        # xx = 2.0 * torch.tensor(K) * torch.ones(batch_size, seq_len,
        #                                         input_size)
        # xx = xx.transpose(0, 1)
        # print(xx[0] @ self.weight_ih[0].to_dense().T)
        # print(torch.sparse.mm(xx[0], ww[0].T))

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
            selected_indices = torch.randperm(hidden_size)[:K]
            indices.extend([[i, j] for j in selected_indices])
            values.extend(
                [self.JII / torch.sqrt(torch.tensor(K, dtype=torch.float32))] *
                K)

        indices = torch.tensor(indices).t()
        values = torch.tensor(values, dtype=torch.float32)
        sparse_weight = torch.sparse.FloatTensor(
            indices, values, torch.Size([hidden_size, hidden_size]))
        return sparse_weight

    def transfer_function(self, total_input):
        return F.relu(total_input) # should we care about positive rates?
        # return torch.ones(x.shape) +
        # return torch.tanh(total_input)

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
                if layer == 0:  # x[t] @ self.weight_ih[layer].T
                    h_t[layer] = self.transfer_function(
                        torch.sparse.mm(self.weight_ih[layer], x[t].T).T +
                        torch.sparse.mm(self.weight_hh[layer],
                                        h_t_minus_1[layer].T).T)
                else:
                    h_t[layer] = self.transfer_function(
                        torch.sparse.mm(self.weight_hh[layer], \
                                        h_t[layer - 1].T).T +
                        torch.sparse.mm(self.weight_hh[layer],
                                        h_t_minus_1[layer].T).T)

                # h_t[layer] = self.transfer_function(
                #     x[t] @ self.weight_ih[layer].T +
                #     self.bias_ih[layer] + torch.sparse.mm(
                #         self.weight_hh[layer], h_t_minus_1[layer].T).T +
                #     self.bias_hh[layer])

                # h_t[layer] = self.transfer_function(
                #     torch.sparse.mm(self.weight_hh[layer], h_t[layer -
                #                                                1].T).T +
                #     self.bias_ih[layer] + torch.sparse.mm(
                #         self.weight_hh[layer], h_t_minus_1[layer].T).T +
                #     self.bias_hh[layer])
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
    rnn = CustomRNN(input_size,
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

    # plt.hist(h_n.detach().numpy().squeeze(), 25)

    rates = output.detach().numpy().squeeze()
    #.mean(axis=1)
    # plt.plot(rates[:, :10])
    fig, ax = plt.subplots()
    ri = rates.mean(axis=0)
    print(ri.min(), ri.max(), sum(ri <= 0))
    ax.hist(ri, 25)
    ax.set_xscale('symlog')
    plt.title('rate distr: is this balanced??')
    plt.figure()
    plt.plot(rates.mean(axis=1))
    plt.title('network avg rate')
    print(sum(rates.mean(axis=1) <= 0))
    plt.show()
