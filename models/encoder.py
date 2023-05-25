import torch
import torch.nn as nn

from models.cnn_layer import CNNLayer

class CNN2DShapesEncoder(nn.Module):
    def __init__(self, config):
        super(CNN2DShapesEncoder, self).__init__()
        modules = []
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states
        self.num_sampling = config.num_sampling

        #Design Encoder Factor-VAE ref
        modules.append(CNNLayer(in_channels=1, out_channels=32))
        modules.append(CNNLayer(in_channels=32, out_channels=32))
        modules.append(CNNLayer(in_channels=32, out_channels=64))
        modules.append(CNNLayer(in_channels=64, out_channels=64))
        self.hidden_layers = nn.ModuleList(modules)

        self.dense = nn.Linear(config.dense_dim[0], config.dense_dim[1])
        self.mu = nn.Linear(config.dense_dim[1], self.latent_dim)
        self.logvar = nn.Linear(config.dense_dim[1], self.latent_dim)

    def forward(self, input):
        all_hidden_states = ()

        output = input
        if self.hidden_states:
            all_hidden_states = all_hidden_states + (output,)
        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)
            if self.hidden_states:
                all_hidden_states = all_hidden_states + (output,)
        # output = torch.flatten(output, start_dim=1)
        output = self.dense(output.contiguous().view(output.size(0), -1))  # 4-D tensor: [Batch, *] --> 2-D tensor: [Batch, latent dim]
        mu = self.mu(output)  #[Batch, latent dim]
        logvar = self.logvar(output)  #[Batch, latent dim]

        z = self.reparameterization(mu, logvar)
        outputs = (z, mu, logvar,) + (all_hidden_states,)  # (z, mu, logvar, (outputs))
        return outputs

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.num_sampling == 1:

            eps = torch.randn_like(logvar)
            z = mu + std * eps
            return z
        else:
            batch, dim = std.size()
            eps = torch.randn(size=(self.num_sampling, batch, dim))
            z = mu + std * eps

            return z

class CNN3DShapesEncoder(CNN2DShapesEncoder):
    def __init__(self, config):
        super(CNN3DShapesEncoder, self).__init__(config)
        modules = []
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states
        self.num_sampling = config.num_sampling

        # Design Encoder Factor-VAE ref
        modules.append(CNNLayer(in_channels=3, out_channels=32))
        modules.append(CNNLayer(in_channels=32, out_channels=32))
        modules.append(CNNLayer(in_channels=32, out_channels=64))
        modules.append(CNNLayer(in_channels=64, out_channels=64))
        self.hidden_layers = nn.ModuleList(modules)

        self.dense = nn.Linear(config.dense_dim[0], config.dense_dim[1])
        self.mu = nn.Linear(config.dense_dim[1], self.latent_dim)
        self.logvar = nn.Linear(config.dense_dim[1], self.latent_dim)

# Lie Group based models implemented https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch
# Modified original code
class CNN2DShapesLieEncoder(CNN2DShapesEncoder):
    def __init__(self, config):
        super(CNN2DShapesLieEncoder, self).__init__(config)

        #modules = []
        self.to_means = nn.ModuleList([])
        self.to_logvar = nn.ModuleList([])

        self.subgroup_sizes_ls = config.subgroup_sizes_ls
        self.subspace_sizes_ls = config.subspace_sizes_ls
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states
        self.num_sampling = config.num_sampling

        self.dense1 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(256, sum(self.subgroup_sizes_ls))
        self.active = nn.Sigmoid()

        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls): #subgroup_size int = latent_dim
            self.to_means.append(
                nn.Sequential(
                    nn.Linear(subgroup_size_i, subgroup_size_i * 4),
                    nn.ReLU(True),
                    nn.Linear(subgroup_size_i * 4, self.subspace_sizes_ls[i]),
                ))
            self.to_logvar.append(
                nn.Sequential(
                    nn.Linear(subgroup_size_i, subgroup_size_i * 4),
                    nn.ReLU(True),
                    nn.Linear(subgroup_size_i * 4, self.subspace_sizes_ls[i]),
                ))
        #self.hidden_layers = nn.ModuleList(modules)

    def forward(self, input):

        group_feats = input
        for i, hidden_layer in enumerate(self.hidden_layers):
            group_feats = hidden_layer(group_feats)

        group_feats = self.dense1(group_feats.view(group_feats.size(0), -1)) # 4-D tensor: [Batch, *] --> 2-D tensor: [Batch, latent dim]
        group_feats = self.relu(group_feats)
        group_feats = self.dense2(group_feats)
        group_feats = self.active(group_feats)

        b_idx = 0
        means_ls, logvars_ls = [], []

        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            e_idx = b_idx + subgroup_size_i
            means_ls.append(self.to_means[i](group_feats[:, b_idx:e_idx]))
            logvars_ls.append(self.to_logvar[i](group_feats[:, b_idx:e_idx]))
            b_idx = e_idx
        mu = torch.cat(means_ls, dim=-1)
        logvar = torch.cat(logvars_ls, dim=-1)
        z = self.reparameterization(mu, logvar)
        outputs = (z, mu, logvar, group_feats) # z: after main encoder

        return outputs

# Lie Group based models implemented https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch
# Modified original code
class CNN3DShapesLieEncoder(CNN3DShapesEncoder):
    def __init__(self, config):
        super(CNN3DShapesLieEncoder, self).__init__(config)

        #modules = []
        self.to_means = nn.ModuleList([])
        self.to_logvar = nn.ModuleList([])

        self.subgroup_sizes_ls = config.subgroup_sizes_ls
        self.subspace_sizes_ls = config.subspace_sizes_ls
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states
        self.num_sampling = config.num_sampling

        self.dense1 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(256, sum(self.subgroup_sizes_ls))
        self.active = nn.Sigmoid()

        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls): #subgroup_size int = latent_dim
            self.to_means.append(
                nn.Sequential(
                    nn.Linear(subgroup_size_i, subgroup_size_i * 4),
                    nn.ReLU(True),
                    nn.Linear(subgroup_size_i * 4, self.subspace_sizes_ls[i]),
                ))
            self.to_logvar.append(
                nn.Sequential(
                    nn.Linear(subgroup_size_i, subgroup_size_i * 4),
                    nn.ReLU(True),
                    nn.Linear(subgroup_size_i * 4, self.subspace_sizes_ls[i]),
                ))

    def forward(self, input):

        group_feats = input
        for i, hidden_layer in enumerate(self.hidden_layers):
            group_feats = hidden_layer(group_feats)

        group_feats = self.dense1(group_feats.reshape(group_feats.size(0), -1)) # 4-D tensor: [Batch, *] --> 2-D tensor: [Batch, latent dim]
        group_feats = self.relu(group_feats)
        group_feats = self.dense2(group_feats)
        group_feats = self.active(group_feats)

        b_idx = 0
        means_ls, logvars_ls = [], []

        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            e_idx = b_idx + subgroup_size_i
            means_ls.append(self.to_means[i](group_feats[:, b_idx:e_idx]))
            logvars_ls.append(self.to_logvar[i](group_feats[:, b_idx:e_idx]))
            b_idx = e_idx
        mu = torch.cat(means_ls, dim=-1)
        logvar = torch.cat(logvars_ls, dim=-1)
        z = self.reparameterization(mu, logvar)
        outputs = (z, mu, logvar, group_feats) # z: after main encoder

        return outputs