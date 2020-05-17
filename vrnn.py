import torch
import torch.nn as nn
import torch.utils
from torch.autograd import Variable

from config import device

class VRNN(nn.Module):
    """
    Implementation of Variational Recurrent Neural Network
    (VRNN) from https://arxiv.org/abs/1506.02216

    Code adapted from https://github.com/emited/VariationalRecurrentNeuralNetwork
    and https://github.com/p0werHu/VRNN
    """
    def __init__(self, x_dim , h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # feature extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        # prior: input h output mu, sigma
        self.prior_fea = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # encoder: input: phi(x), h
        self.encoder = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU()
        )
        # VRE regard mean values sampled from z as the output
        self.encoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        self.encoder_std = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )

        # decoder: input phi(z), h
        self.decoder_fea = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU()
        )
        self.decoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )
        self.decoder_std = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Softplus()
        )

        # recurrence step
        self.rnn = nn.GRUCell(h_dim + h_dim, h_dim)

    def forward(self, x, _max, _min):
        """
        Input is a batch of trajectories
        """
        kld_loss = 0
        mse_loss = 0
        h = torch.zeros([x.shape[0], self.h_dim]).to(device)

        for t in range(x.shape[1]):
            # feature extractor 
            phi_x = self.phi_x(x[:, t, :])

            # prior
            prior_fea = self.prior_fea(h)
            prior_mean = self.prior_mean(prior_fea)
            prior_std = self.prior_std(prior_fea)

            # encoder
            encoder_fea = self.encoder(torch.cat([phi_x, h], dim=1))
            encoder_mean = self.encoder_mean(encoder_fea)
            encoder_std = self.encoder_std(encoder_fea)

            # sampling and reparametrization
            z_t = self._reparametrized_sample(encoder_mean, encoder_std)
            phi_z_t = self.phi_z(z_t)

            # decoder
            decoder_fea = self.decoder_fea(torch.cat([phi_z_t, h], dim=1))
            decoder_mean = self.decoder_mean(decoder_fea)
            decoder_std = self.decoder_std(decoder_fea)

            # recurrence step
            h = self.rnn(torch.cat([phi_x, phi_z_t], dim=1), h)

            # compute loss
            kld_loss += self._kld_gauss(encoder_mean, encoder_std, prior_mean, prior_std)
            mse_loss += self._mse((decoder_mean * _max) + _min, (x[:, t, :] * _max) + _min)

        return kld_loss, mse_loss

    def sample(self, seq_len):
        _sample = torch.zeros(seq_len, self.x_dim)
        h = torch.zeros(1, self.h_dim).to(device)

        for t in range(seq_len):
            # prior
            prior_fea_ = self.prior_fea(h)
            prior_means_ = self.prior_mean(prior_fea_)
            prior_var_ = self.prior_std(prior_fea_)

            # decoder
            z_t = self._reparametrized_sample(prior_means_, prior_var_)
            phi_z_t = self.phi_z(z_t)
            decoder_fea_ = self.decoder_fea(torch.cat([phi_z_t, h], dim=1))
            decoder_means_ = self.decoder_mean(decoder_fea_)

            phi_x_t = self.phi_x(decoder_means_)
            
            # rnn
            h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1), h)

            _sample[t] = decoder_means_.detach()
        
        return _sample

    def _reparametrized_sample(self, mean, std):
        s = torch.FloatTensor(std.size()).normal_()
        s = Variable(s)
        s = s.to(device)
        return s.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        return -torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))

    def _mse(self, x, reconstructed_x):
        return ((x - reconstructed_x) ** 2).mean()
