import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Constants import DATA_CLASSES
from models.ipe_transformation import Invert_Equiv_Func#, Invert_Equiv_Func_without_symmetric

class MIPET_BetaTCVAE(nn.Module):
    def __init__(self, config):
        super(MIPET_BetaTCVAE, self).__init__()
        self.dataset_size = config.dataset_size
        self.num_inv_equ = config.num_inv_equ
        # self.prob = config.prob
        encoder, decoder = DATA_CLASSES[config.dataset]
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.inv_equ = nn.ModuleList([])
        for _ in range(self.num_inv_equ):
            self.inv_equ.append(Invert_Equiv_Func(config))

    def forward(self, input, loss_fn):
        new_zs = []
        kl_err, reg_err = 0.0, 0.0

        result = {'elbo': {}, 'obj': {}, 'id': {}}
        batch = input.size(0)

        enc_output = self.encoder(input)
        z, mu, logvar = enc_output[0], enc_output[1], enc_output[2]

        ori_kld_err = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1)
        mi, tc, kld = self.betatcloss(z, mu, logvar)

        std = torch.exp(0.5 * logvar)
        eps = (z - mu) / std

        for i, inv_equ in enumerate(self.inv_equ):
            new_z, _, loss_reg, kl_loss = self.inv_equ[i](z, eps)
            kl_err = kl_err + kl_loss
            reg_err = reg_err + loss_reg
            new_zs.append(new_z)

        new_output = torch.stack(new_zs, dim=0)
        new_output = self.decoder(new_output)
        new_output = list(new_output)
        _, c, h, w = new_output[0].size()
        new_output[0] = torch.mean(new_output[0].view(self.num_inv_equ, batch, c, h, w), dim=0)
        new_output = tuple(new_output)

        # criteria = nn.BCELoss(reduction='sum')
        reconst_err = loss_fn(new_output[0], input) / batch
        reg_err = reg_err.squeeze()  # [Batch, dim]
        kl_err = kl_err.sum(dim=-1).squeeze()  # [Batch, dim]
        reg_err = reg_err.mean()
        reg_err = reg_err + F.mse_loss(kl_err, self.num_inv_equ * ori_kld_err)
        kl_err = kl_err.mean()
        ori_kld_err = ori_kld_err.mean()

        result['obj']['reconst'] = reconst_err.unsqueeze(0)
        result['obj']['kld'] = kld.unsqueeze(0) # Beta-TCVAE
        result['obj']['mi'] = mi.unsqueeze(0) # Beta-TCVAE
        result['obj']['tc'] = tc.unsqueeze(0) # Beta-TCVAE
        result['obj']['reg'] = reg_err.unsqueeze(0)
        result['obj']['origin_kld'] = ori_kld_err.unsqueeze(0)
        output = (result,) + (enc_output,) + (new_output,)
        return output

    def betatcloss(self, z, mu, logvar):
        batch = z.size(0)
        zeros = torch.zeros_like(z)
        logqzx = self.log_density_gaussian(z, mu, logvar).sum(dim=1)
        logpz = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)  # size: batch
        _logqz = self.log_density_gaussian(z.view(batch, 1, -1),
                                           mu.view(1, batch, -1),
                                           logvar.view(1, batch, -1))  # size: (batch, batch, dim)

        stratified_weight = (self.dataset_size - batch + 1) / (self.dataset_size * (batch - 1))
        importance_weights = torch.Tensor(batch, batch).fill_(1 / (batch - 1)).to(z.device)
        importance_weights.view(-1)[::batch] = 1 / self.dataset_size
        importance_weights.view(-1)[1::batch] = stratified_weight
        importance_weights[batch - 2, 0] = stratified_weight
        log_importance_weights = importance_weights.log()
        _logqz += log_importance_weights.view(batch, batch, 1)

        logqz_prod = torch.logsumexp(_logqz, dim=1, keepdim=False).sum(1)  # - math.log(batch)).sum(1)  # size: batch
        logqz = torch.logsumexp(_logqz.sum(2), dim=1, keepdim=False)  # - math.log(batch)

        mi = torch.mean(logqzx - logqz)
        tc_err = torch.mean(logqz - logqz_prod)
        kld_err = torch.mean(logqz_prod - logpz)

        return mi, tc_err, kld_err

    def log_density_gaussian(self, z, mu, logvar):
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * (z - mu) ** 2 * torch.exp(-logvar)
        return log_density

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)

# Ablation for without Equivariant
#class MIPET_BetaTCVAE_without_Symmetric(MIPET_BetaTCVAE):
#    def __init__(self, config):
#        super(MIPET_BetaTCVAE_without_Symmetric, self).__init__(config)
#        self.inv_equ = nn.ModuleList([])
#        for _ in range(self.num_inv_equ):
#            self.inv_equ.append(Invert_Equiv_Func_without_symmetric(config))

# Ablation for without Regularization
class MIPET_BetaTCVAE_without_Regularizer(MIPET_BetaTCVAE):
    def __init__(self, config):
        super(MIPET_BetaTCVAE_without_Regularizer, self).__init__(config)

    def forward(self, input, loss_fn):
        new_zs = []
        kl_err, reg_err = 0.0, 0.0

        result = {'elbo': {}, 'obj': {}, 'id': {}}
        batch = input.size(0)

        enc_output = self.encoder(input)
        z, mu, logvar = enc_output[0], enc_output[1], enc_output[2]

        kl_err = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=-1)
        mi, tc, kld = self.betatcloss(z, mu, logvar)

        std = torch.exp(0.5 * logvar)
        eps = (z - mu) / std

        for i, inv_equ in enumerate(self.inv_equ):
            new_z, _, _, _ = self.inv_equ[i](z, eps)
            new_zs.append(new_z)

        # version 0.2
        new_output = torch.stack(new_zs, dim=0)
        #new_output = torch.cat(new_zs, dim=-1)
        new_output = self.decoder(new_output)
        new_output = list(new_output)
        # for conditional independence
        _, c, h, w = new_output[0].size()
        new_output[0] = torch.mean(new_output[0].view(self.num_inv_equ, batch, c, h, w), dim=0)
        new_output = tuple(new_output)

        reconst_err = loss_fn(new_output[0], input) / batch
        kl_err = kl_err.mean()

        result['obj']['reconst'] = reconst_err.unsqueeze(0)
        result['obj']['kld'] = kld.unsqueeze(0) # Beta-TCVAE
        result['obj']['mi'] = mi.unsqueeze(0) # Beta-TCVAE
        result['obj']['tc'] = tc.unsqueeze(0) # Beta-TCVAE
        result['obj']['reg'] = 0.0 #reg_err.unsqueeze(0)
        result['obj']['origin_kld'] = kld.unsqueeze(0) # ori_kld_err.unsqueeze(0)
        output = (result,) + (enc_output,) + (new_output,)
        return output