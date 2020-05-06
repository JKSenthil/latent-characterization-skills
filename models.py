import torch
from torch import nn
from torch.autograd import Variable

"""
NOTE: this file is not in use
"""

class VRNNCell(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VRNNCell, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, z_dim),
            nn.ELU() # TODO change activation function?
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, x_dim)
        )

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, z_dim),
            nn.Sigmoid()
        )

        self.rnn = nn.GRUCell(x_dim + z_dim, h_dim)
    
    def forward(self, x, hidden):
        z_prior = self.prior(hidden)
        z_infer = self.encoder(torch.cat([x, hidden], dim=1))
        # sampling
        _h = z_prior.shape[1] / 2
        z = Variable(torch.randn(x.size(0), _h)) * z_infer[:,_h:].exp() + z_infer[:,:_h]
        x_out = self.decoder(torch.cat([hidden, z], dim=1))
        next_hidden = self.rnn(torch.cat([x,z], dim=1), hidden)
        return x_out, next_hidden, z_prior, z_infer
    
    def calculate_loss(self, x, hidden):
        x_out, next_hidden, z_prior, z_infer = self.forward(x, hidden)
        # mse loss
        loss1 = self._mse(x, x_out)
        # kl divergence loss
        _h = z_prior.shape[1] / 2
        mu_infer, log_sigma_infer = z_infer[:,:_h], z_infer[:,_h:]
        mu_prior, log_sigma_prior = z_prior[:,:_h], z_prior[:,_h:]
        loss2 = self._kld_gauss(mu_infer, log_sigma_infer, mu_prior, log_sigma_prior)
        return loss1, loss2

    def _mse(self, x, x_out):
        return ((x - x_out) ** 2).mean()
    
    def _kld_gauss(self, mu_infer, log_sigma_infer, mu_prior, log_sigma_prior):
        l = (2*(log_sigma_infer-log_sigma_prior)).exp() \
                + ((mu_infer-mu_prior)/log_sigma_prior.exp())**2 \
                - 2*(log_sigma_infer-log_sigma_prior) - 1
        return 0.5*l.sum(dim=1).mean()

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VRNN, self).__init__()
        self.cell = VRNNCell(x_dim, h_dim, z_dim)
    
    def forward(self, x):
        mse_loss = 0
        kl_loss = 0
        hidden = torch.zeros([x.shape[0], self.h_dim]) #.to(device)

        for t in range(x.shape[1]):
            _x = x[:,t,:]
            x_out, next_hidden, z_prior, z_infer = self.cell(_x, hidden)
            _l1, _l2 = self.cell.calculate_loss(_x, hidden)
            mse_loss += _l1
            kl_loss += _l2
        
        return kl_loss, mse_loss
    
    def sample(self):
        pass # TODO