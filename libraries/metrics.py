import torch.nn as nn

# NOTE I will put here all usefull metrics for evaluation like Wasserstein and Smirnoff etc 

class WassersteinMetric(nn.Module):
    
    def __init__(self):
        super(WassersteinMetric, self).__init__()

    def forward(self,u_values, v_values, u_weights=None, v_weights=None):

    #def torch_wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
        # NOTE Wasserstein Metrik
        # https://forkxz.github.io/blog/2024/Wasserstein/

        # Ensure that the input tensors are batched
        assert u_values.dim() == 2 and v_values.dim() == 2, "Input tensors must be 2-dimensional (batch_size, num_values)"

        batch_size, u_size = u_values.shape
        _, v_size = v_values.shape

        # Sort the values
        u_sorter = torch.argsort(u_values, dim=1)
        v_sorter = torch.argsort(v_values, dim=1)

        # Concatenate and sort all values for each batch
        all_values = torch.cat((u_values, v_values), dim=1)
        all_values, _ = torch.sort(all_values, dim=1)
        # Compute differences between successive values
        deltas = torch.diff(all_values, dim=1)

        # Get the respective positions of the values of u and v among the values of both distributions
        all_continue = all_values[:, :-1].contiguous()
        u_cdf_indices = torch.searchsorted(u_values.gather(1, u_sorter).contiguous(), all_continue, right=True)
        v_cdf_indices = torch.searchsorted(v_values.gather(1, v_sorter).contiguous(), all_continue, right=True)

        # Calculate the CDFs of u and v using their weights, if specified
        if u_weights is None:
            u_cdf = u_cdf_indices.float() / u_size
        else:
            u_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(u_weights.gather(1, u_sorter), dim=1)), dim=1)
            u_cdf = u_sorted_cumweights.gather(1, u_cdf_indices) / u_sorted_cumweights[:, -1].unsqueeze(1)

        if v_weights is None:
            v_cdf = v_cdf_indices.float() / v_size
        else:
            v_sorted_cumweights = torch.cat((torch.zeros((batch_size, 1)), torch.cumsum(v_weights.gather(1, v_sorter), dim=1)), dim=1)
            v_cdf = v_sorted_cumweights.gather(1, v_cdf_indices) / v_sorted_cumweights[:, -1].unsqueeze(1)

        return torch.sum(torch.abs(u_cdf - v_cdf) * deltas, dim=1)