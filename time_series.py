import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BalRNN(nn.Module):
    '''one pop bal network with training on sqrt(K) weights'''

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 K=10,
                 JI0=0.4,
                 JII=-0.1):
        super(BalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.K = K
        self.JI0 = JI0
        self.JII = JII
        #
        self.history = []
        # ff
        self.weight_ih = nn.ParameterList([
            nn.Parameter(
                self.initialize_ff_weight(input_size, hidden_size, K, JII))
            for _ in range(num_layers)
        ])

        # Initialize hidden-to-hidden recurrent weights with masking
        self.weight_hh = nn.ParameterList([
            nn.Parameter(self.initialize_masked_weight(hidden_size, K, JII))
            for _ in range(num_layers)
        ])

        # Linear layer to map from hidden_size to input_size
        self.output_layer = nn.Linear(hidden_size, input_size)

    def initialize_ff_weight(self, input_size, hidden_size, K, JI0):
        indices = []
        values = []
        for i in range(hidden_size):
            tmp_K = np.where(np.random.rand(input_size) <= K / input_size)[0]
            selected_indices = torch.tensor(tmp_K)
            num_selected_indices = selected_indices.shape[0]
            indices.extend([[i, j] for j in selected_indices])
            values.extend(
                [self.JI0 / torch.sqrt(torch.tensor(K, dtype=torch.float32))] *
                num_selected_indices)  # list mult

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
            tmp_K = np.where(np.random.rand(hidden_size) <= K / hidden_size)[0]
            selected_indices = torch.tensor(tmp_K)
            num_selected_indices = selected_indices.shape[0]

            mask_indices.extend([[i, j] for j in selected_indices])
            mask_values.extend(
                [J / torch.sqrt(torch.tensor(K, dtype=torch.float32))] *
                num_selected_indices)

            trainable_indices = selected_indices[:int(K**0.5)]
            trainable_values = [
                J / torch.sqrt(torch.tensor(int(K**0.5), dtype=torch.float32))
            ] * int(K**0.5)
            mask_values[-int(K**0.5):] = trainable_values

            trainable_mask = torch.zeros(hidden_size)
            trainable_mask[trainable_indices] = 1
            trainable_mask = trainable_mask.unsqueeze(1)
            trainable_mask = trainable_mask.repeat(1, hidden_size)
            trainable_mask = trainable_mask.reshape(-1)

        mask_indices = torch.tensor(mask_indices).t()
        mask_values = torch.tensor(mask_values, dtype=torch.float32)
        masked_weight = torch.sparse.FloatTensor(
            mask_indices, mask_values, torch.Size([hidden_size, hidden_size]))

        trainable_mask = torch.tensor(trainable_mask,
                                      dtype=torch.bool).clone().detach()
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
        # Apply the linear layer to map to input_size
        output = self.output_layer(output)
        return output, h_t


def train_network(model, data, num_epochs=100, learning_rate=0.001):
    """
    Train the BalRNN on the given time series data.

    Args:
        model (BalRNN): The recurrent neural network model to be trained.
        data (np.ndarray): The time series data for training of shape (num_samples, seq_len, num_features).
        num_epochs (int): The number of epochs for training.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        list: Training loss history.
    """
    # Convert data to PyTorch tensor
    data = torch.tensor(data, dtype=torch.float32)

    # Define the loss function
    criterion = nn.MSELoss()

    # Separate the parameters into dense and sparse
    dense_params = []
    sparse_params = []
    for name, param in model.named_parameters():
        if param.is_sparse:
            sparse_params.append(param)
        else:
            dense_params.append(param)

    print(sparse_params)

    # Define separate optimizers for dense and sparse parameters
    optimizer_dense = optim.Adam(dense_params, lr=learning_rate)
    optimizer_sparse = optim.SparseAdam(sparse_params, lr=learning_rate)

    # Training loop
    loss_history = []
    for epoch in range(num_epochs):
        model.train()

        # Zero the gradients for both optimizers
        optimizer_dense.zero_grad()
        optimizer_sparse.zero_grad()

        # Forward pass
        output, _ = model(data)

        # Compute the loss
        loss = criterion(output, data)

        # Backward pass and optimization
        loss.backward()
        optimizer_dense.step()
        optimizer_sparse.step()

        # Record the loss
        loss_history.append(loss.item())

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return loss_history


def generate_time_series(seq_len, num_features, num_samples=1):
    """
    Generate a synthetic time series.

    Args:
        seq_len (int): The length of the time series.
        num_features (int): The number of features in the time series.
        num_samples (int): The number of samples to generate.

    Returns:
        np.ndarray: The generated time series data of shape (num_samples, seq_len, num_features).
    """
    t = np.linspace(0, 10, seq_len)
    time_series = np.zeros((num_samples, seq_len, num_features))

    for i in range(num_features):
        freq = np.random.uniform(0.1, 1.0)
        phase = np.random.uniform(0, 2 * np.pi)
        time_series[:, :, i] = np.sin(2 * np.pi * freq * t + phase)

    return time_series


if __name__ == '__main__':
    input_size = 250  # num of ff neurons
    hidden_size = 1000  # num of neurons in the balanced network
    num_layers = 1  # 1 layer of balanced network
    batch_size = 10
    seq_len = 100  # simulation time
    K = 100
    JI0 = 1.0
    JII = -0.001

    # Generate synthetic time series data
    time_series_data = generate_time_series(seq_len,
                                            input_size,
                                            num_samples=batch_size)

    # Initialize the model
    rnn = BalRNN(input_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 K=K,
                 JI0=JI0,
                 JII=JII)

    # Train the model
    loss_history = train_network(rnn,
                                 time_series_data,
                                 num_epochs=100,
                                 learning_rate=0.001)

    # Plot the training loss
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
