import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Constants import DATA_CLASSES
from models.ipe_transformation import Invert_Equiv_Func#, Invert_Equiv_Func_without_symmetric

class MIPET_VAE(nn.Module):

    def __init__(self, config):
        super(MIPET_VAE, self).__init__()
        self.num_inv_equ = config.num_inv_equ
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

        std = torch.exp(0.5 * logvar)
        eps = (z - mu) / std
        for i, inv_equ in enumerate(self.inv_equ):
            new_z, _, loss_reg, kl_loss = self.inv_equ[i](z, eps)
            kl_err = kl_err + kl_loss
            reg_err = reg_err + loss_reg
            new_zs.append(new_z)

        new_output = torch.stack(new_zs, dim=0)
        # new_output = torch.cat(new_zs, dim=-1)
        new_output = self.decoder(new_output)
        new_output = list(new_output)
        # for conditional independence
        _, c, h, w = new_output[0].size()
        new_output[0] = torch.mean(new_output[0].view(self.num_inv_equ, batch, c, h, w), dim=0)
        new_output = tuple(new_output)

        # criteria = nn.BCELoss(reduction='sum')
        reconst_err = loss_fn(new_output[0], input) / batch

        reg_err = reg_err.squeeze()  # [Batch]
        kl_err = kl_err.sum(dim=-1, keepdim=False).squeeze()  # [Batch]
        reg_err = reg_err + F.mse_loss(kl_err, self.num_inv_equ * ori_kld_err)
        reg_err = reg_err.mean()
        kl_err = kl_err.mean()
        ori_kld_err = ori_kld_err.mean()

        result['obj']['reconst'] = reconst_err.unsqueeze(0)
        result['obj']['kld'] = kl_err.unsqueeze(0)
        result['obj']['reg'] = reg_err.unsqueeze(0)
        result['obj']['origin_kld'] = ori_kld_err.unsqueeze(0)
        output = (result,) + (enc_output,) + (new_output,)
        return output

    def kl_divergence(self, pdf_z, pdf_eps):  # (Batach, 1, latent_dim)
        kl = (pdf_z * torch.log(pdf_z / pdf_eps)).sum(-1)
        return kl

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)

# Ablation for without Equivariant
#class MIPET_CVAE_without_Symmetric(MIPET_VAE):
#    def __init__(self, config):
#        super(MIPET_CVAE_without_Symmetric, self).__init__(config)
#        self.inv_equ = nn.ModuleList([])
#        for _ in range(self.num_inv_equ):
#            self.inv_equ.append(Invert_Equiv_Func_without_symmetric(config))

# Ablation for without Regularization
class MIPET_VAE_without_Regularizer(MIPET_VAE):
    def __init__(self, config):
        super(MIPET_VAE_without_Regularizer, self).__init__(config)