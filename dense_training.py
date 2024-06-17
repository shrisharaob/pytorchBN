import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

import matplotlib.pyplot as plt


class BalRNN(nn.Module):
    '''one pop bal network with training on dense weights'''

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

        # ff
        self.weight_ih = nn.ParameterList([
            nn.Parameter(
                self.initialize_ff_weight(input_size, hidden_size, K, JI0))
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
        weight = torch.zeros(hidden_size, input_size)
        for i in range(hidden_size):
            tmp_K = np.where(np.random.rand(input_size) <= K / input_size)[0]
            selected_indices = torch.tensor(tmp_K)
            weight[i, selected_indices] = JI0 / torch.sqrt(
                torch.tensor(K, dtype=torch.float32))
        return weight

    # def initialize_dense_weight(self, hidden_size, K, J):
    #     weight = torch.zeros(hidden_size, hidden_size)
    #     for i in range(hidden_size):
    #         tmp_K = np.where(np.random.rand(hidden_size) <= K / hidden_size)[0]
    #         selected_indices = torch.tensor(tmp_K)
    #         weight[i, selected_indices] = J / torch.sqrt(
    #             torch.tensor(K, dtype=torch.float32))
    #     return weight

    def initialize_masked_weight(self, hidden_size, K, J):
        mask_indices = []
        mask_values = []

        for i in range(hidden_size):
            # Randomly select K indices for connections
            # selected_indices = torch.randperm(hidden_size)[:K]

            tmp_K = np.where(np.random.rand(hidden_size) <= K / hidden_size)[0]
            selected_indices = torch.tensor(tmp_K)
            num_selected_indices = selected_indices.shape[0]

            # Extend mask_indices with the selected indices for the current unit
            mask_indices.extend([[i, j] for j in selected_indices])

            # Set values for these connections to J/sqrt(K)
            mask_values.extend(
                [J / torch.sqrt(torch.tensor(K, dtype=torch.float32))] *
                num_selected_indices)

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
        return masked_weight.to_dense()

    def transfer_function(self, total_input):
        return F.relu(total_input)

    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Ensure (seq_len, batch_size, input_size)
            # x = x.transpose(1, 0, 2)
        # seq_len, batch_size, _ = x.size()
        seq_len, batch_size, _ = x.shape
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        h_t_minus_1 = h_0
        h_t = h_0.clone()
        output = []
        # constant ff_input
        const_ff_input = torch.ones(self.hidden_size) * torch.sqrt(
            torch.tensor(self.K))
        for t in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0:
                    h_t[layer] = self.transfer_function(
                        const_ff_input +
                        torch.mm(x[t], self.weight_ih[layer].T) +
                        torch.mm(h_t_minus_1[layer], self.weight_hh[layer].T))
                else:
                    h_t[layer] = self.transfer_function(
                        const_ff_input +
                        torch.mm(h_t[layer - 1], self.weight_hh[layer].T) +
                        torch.mm(h_t_minus_1[layer], self.weight_hh[layer].T))
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

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    loss_history = []
    for epoch in range(num_epochs):
        model.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output, _ = model(data)

        # Compute the loss
        loss = criterion(output, data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Record the loss
        loss_history.append(loss.item())

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return loss_history


def generate_time_series(seq_len, num_features, K, num_samples=1):
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

    return time_series / np.sqrt(K)


def plot_time_series_comparisons(model, time_series_data, num_samples=3):
    """
    Plot comparisons between ground truth time series and model predictions.

    Args:
        model (BalRNN): Trained recurrent neural network model.
        time_series_data (np.ndarray): Generated time series data of shape (num_samples, seq_len, num_features).
        num_samples (int): Number of samples to plot comparisons for.
    """
    model.eval()
    with torch.no_grad():
        # Select random samples
        indices = np.random.choice(len(time_series_data),
                                   num_samples,
                                   replace=False)
        for idx in indices:
            sample = time_series_data[idx]
            input_seq = torch.tensor(sample, dtype=torch.float32).unsqueeze(
                0)  # Add batch dimension
            # Predictions
            predicted_seq, _ = model(input_seq)
            predicted_seq = predicted_seq.squeeze().detach().numpy()
            # Plot
            plt.figure()
            for i in [0, 1, 5]:
                #range(sample.shape[1]):  # Plot each feature
                plt.plot(sample[:, i],
                         label=f'Feature {i+1} (Ground Truth)',
                         linestyle='--')
                plt.plot(predicted_seq[:, i],
                         label=f'Feature {i+1} (Predicted)')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title(f'Time Series Comparison (Sample {idx+1})')
            plt.legend()
            plt.show()


# def plot_time_series_comparison(gt_time_series, pred_time_series, num_samples=3):
#     """
#     Plot a comparison between the ground truth and predicted time series.

#     Args:
#         gt_time_series (np.ndarray): Ground truth time series data of shape (num_samples, seq_len, num_features).
#         pred_time_series (np.ndarray): Predicted time series data of shape (num_samples, seq_len, num_features).
#         num_samples (int): Number of samples to plot.
#     """
#     num_features = min(3, gt_time_series.shape[2])  # Limit to at most 3 features
#     for sample_idx in range(num_samples):
#         plt.figure(figsize=(12, 4))
#         for feature_idx in range(num_features):
#             plt.subplot(1, num_features, feature_idx + 1)
#             plt.plot(gt_time_series[sample_idx, :, feature_idx], label='Ground Truth')
#             plt.plot(pred_time_series[sample_idx, :, feature_idx], label='Predicted')
#             plt.xlabel('Time')
#             plt.ylabel('Value')
#             plt.title(f'Feature {feature_idx + 1}')
#             plt.legend()
#         plt.suptitle(f'Time Series Comparison - Sample {sample_idx + 1}')
#         plt.tight_layout()
#         plt.show()


def plot_individual_feature(gt_time_series,
                            pred_time_series,
                            feature_idx=0,
                            num_samples=3):
    """
    Plot the ground truth and predicted time series for a specific feature.

    Args:
        gt_time_series (np.ndarray): Ground truth time series data of shape (num_samples, seq_len, num_features).
        pred_time_series (np.ndarray): Predicted time series data of shape (num_samples, seq_len, num_features).
        feature_idx (int): Index of the feature to plot.
        num_samples (int): Number of samples to plot.
    """
    for sample_idx in range(num_samples):
        plt.figure()
        plt.plot(gt_time_series[sample_idx, :, feature_idx],
                 label='Ground Truth')
        plt.plot(pred_time_series[sample_idx, :, feature_idx],
                 label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Feature {feature_idx + 1} - Sample {sample_idx + 1}')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    input_size = 250  # num of ff neurons
    hidden_size = 1000  # num of neurons in the balanced network
    num_layers = 1  # 1 layer of balanced network
    batch_size = 5
    seq_len = 100  # simulation time
    K = 100
    JI0 = 1.0
    JII = -0.001

    # Generate synthetic time series data
    time_series_data = generate_time_series(seq_len,
                                            input_size,
                                            num_samples=batch_size,
                                            K=K)

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
                                 num_epochs=50,
                                 learning_rate=0.001)

    # Plot the training loss

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plot_time_series_comparisons(rnn, time_series_data)
