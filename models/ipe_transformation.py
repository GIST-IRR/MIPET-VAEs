import torch
import torch.nn as nn
import torch.nn.functional as F


# Invertible and Equivariant Function
class Invert_Equiv_Func(nn.Module):

    def __init__(self, config):
        super(Invert_Equiv_Func, self).__init__()

        self.latent_dim = config.latent_dim

        # symmetric matrix for invertible and equivariant function
        self.w1 = nn.Parameter(torch.Tensor(config.latent_dim, config.latent_dim))
        self.w2 = nn.Parameter(torch.Tensor(config.latent_dim, config.latent_dim))
        self.subobj = EF_conversion(config)  # KL estimator class

    # latent \Tilde{z} and KL estimator
    def forward(self, latent_z, eps):
        # mapping latent z to exponential family through invertible and equivariant function
        symmetric1 = 0.5 * (self.w1 + self.w1.transpose(-1, -2))
        symmetric2 = 0.5 * (self.w2 + self.w2.transpose(-1, -2))

        exp_w1 = torch.matrix_exp(symmetric1)
        exp_w2 = torch.matrix_exp(symmetric2)
        exp_w = torch.mm(exp_w1, exp_w2)

        new_z = torch.matmul(latent_z, exp_w)
        new_eps = torch.matmul(eps, exp_w)

        # Estimate PDF of exponential family & regularizer
        loss_reg, kl_loss = self.subobj(new_z, new_eps)

        return new_z, new_eps, loss_reg, kl_loss

    def update(self):
        self.subobj.update()


class EF_conversion(nn.Module):

    def __init__(self, config):
        super(EF_conversion, self).__init__()
        # self.mask = config.mask
        self.dim = config.latent_dim
        self.bayesian = config.bayesian
        # self._lambda = config._lambda
        self.z_parameter_generator = sub_layers(input_dim=self.dim, output_dim=self.dim)
        self.eps_parameter_generator = sub_layers(input_dim=self.dim, output_dim=self.dim)
        self.suff_stat = sub_layers(input_dim=self.dim, output_dim=self.dim)  # T
        self.log_norm = log_norm(input_dim=self.dim, output_dim=self.dim) # A
        # prior observation
        self.nu = sub_layers(input_dim=self.dim, output_dim=self.dim)
        self.post_carrier = sub_layers(input_dim=2 * self.dim, output_dim=2 * self.dim)
        self.lambda_ = nn.Linear(self.dim, self.dim)

    def forward(self, z, eps):
        z, eps = z.unsqueeze(1), eps.unsqueeze(1)  # [Batch, 1, latent dim]
        theta_z, theta_eps = self.z_parameter_generator(z), self.eps_parameter_generator(eps)  # [Batch, 1, theta dim]

        kl_err, log_norm_z, log_norm_eps = self.kl_loss(theta_z, theta_eps)
        ef_err = self.ef_loss(z, theta_z, theta_eps, log_norm_z, log_norm_eps, kl_err)

        return ef_err, kl_err  # pdf_z, pdf_eps,

    def ef_loss(self, z, theta_z, theta_eps, log_norm_z, log_norm_eps, kl_err):
        batch = z.size(0)

        nabla_z_kl_err = -self.nabla_z(log_norm_z) + self.z_parameter_generator(
            self.log_norm(log_norm_z * (1 - log_norm_z)))  # [Batch, 1, theta dim]
        nabla_z_kl_err = nabla_z_kl_err + torch.matmul(theta_z.transpose(-2, -1),
                                                       self.log_norm(self.nabla_z(log_norm_z) * (1 - log_norm_z))).sum(
            dim=-2, keepdim=True)  # [Batch, 1, theta dim]
        nabla_z_kl_err = nabla_z_kl_err - torch.matmul(theta_z.transpose(-2, -1),
                                                       self.log_norm(self.nabla_z(log_norm_z) * log_norm_z)).sum(dim=-2,
                                                                                                                 keepdim=True)
        nabla_z_kl_err = self.lambda_(nabla_z_kl_err)

        nabla_eps_kl_err = self.nabla_eps(log_norm_eps) - self.eps_parameter_generator(
            self.log_norm(log_norm_eps * (1 - log_norm_eps)))
        nabla_eps_kl_err = nabla_eps_kl_err - torch.matmul(theta_eps.transpose(-2, -1), self.log_norm(
            self.nabla_eps(log_norm_eps) * (1 - log_norm_eps))).sum(dim=-2, keepdim=True)
        nabla_eps_kl_err = nabla_eps_kl_err + torch.matmul(theta_eps.transpose(-2, -1), self.log_norm(
            self.nabla_eps(log_norm_eps) * log_norm_eps)).sum(dim=-2, keepdim=True)
        nabla_eps_kl_err = self.lambda_(nabla_eps_kl_err)

        # self.nu could be changed
        posterior = self.suff_stat(z).sum(dim=0, keepdim=True) + torch.matmul(self.nu(theta_eps).transpose(-2, -1),
                                                                              theta_eps).sum(dim=-2,
                                                                                             keepdim=True)  # [Batch, 1, theta dim]
        posterior = torch.matmul(theta_z.transpose(-2, -1), posterior).sum(dim=-2,
                                                                           keepdim=True)  # [Batch, 1, theta dim]
        posterior = posterior - self.log_norm(theta_z)  # [Batch, 1, theta dim]

        nabla_z_posterior = self.z_parameter_generator(self.suff_stat(z).sum(dim=0, keepdim=True)) + self.suff_stat(
            theta_z) * batch  # [Batch, 1, theta dim]
        nabla_z_posterior = nabla_z_posterior - self.nabla_z(log_norm_z)

        nabla_eps_posterior = 2 * torch.matmul(theta_z.transpose(-2, -1), theta_eps).sum(dim=-2,
                                                                                         keepdim=True)

        nabla_z_obj = torch.norm(nabla_z_kl_err + nabla_z_posterior, dim=-1)  # [Batch, 1]
        nabla_eps_obj = torch.norm(nabla_eps_kl_err + nabla_eps_posterior, dim=-1)  # [Batch, 1]
        nabla_lambda_obj = torch.norm(kl_err, dim=-1)  # [Batch, 1]
        return nabla_z_obj + nabla_eps_obj + nabla_lambda_obj

    def nabla_z(self, log_norm_z):  # for derivative of KLD
        output = log_norm_z * (1 - log_norm_z)  # [Batch, 1, theta dim]
        # pdb.set_trace()
        output = torch.matmul(output, self.log_norm.one_matrix())
        output = self.log_norm(self.z_parameter_generator(output))  # [Batch, 1, theta dim]
        return output

    def nabla_eps(self, log_norm_eps):  # for derivative of KLD
        output = log_norm_eps * (1 - log_norm_eps)  # [Batch, 1, theta dim]
        output = torch.matmul(output, self.log_norm.one_matrix())
        output = self.log_norm(self.eps_parameter_generator(output))  # [Batch, 1, theta dim]
        return output

    def kl_loss(self, theta_z, theta_eps):
        log_norm_z = F.sigmoid(self.log_norm(theta_z))  # [Batch, 1, latent dim]: A(theta{z})
        log_norm_eps = F.sigmoid(self.log_norm(theta_eps))  # [Batch, 1, latent dim]: A(theta_{eps})

        deriv_log_norm_z = torch.matmul(theta_z.transpose(-2, -1),
                                        log_norm_z * (1 - log_norm_z))  # [Batch, theta dim, theta dim]
        deriv_log_norm_z = torch.matmul(deriv_log_norm_z, self.log_norm.one_matrix())  # [Batch, theta dim, theta dim]
        deriv_log_norm_eps = torch.matmul(theta_eps.transpose(-2, -1), log_norm_eps * (1 - log_norm_eps))
        deriv_log_norm_eps = torch.matmul(deriv_log_norm_eps, self.log_norm.one_matrix())

        kl_err = log_norm_eps - deriv_log_norm_eps.sum(dim=-2, keepdim=True) - log_norm_z + deriv_log_norm_z.sum(dim=-2,
                                                                                                                 keepdim=True)
        return kl_err, log_norm_z, log_norm_eps

    def update(self):
        self.log_norm.update()  # update gradient


class log_norm(nn.Module):  # For sparse log_norm
    def __init__(self, input_dim, output_dim):  # ,mask
        super(log_norm, self).__init__()
        # self.mask = mask
        self.w1 = nn.Parameter(torch.Tensor(input_dim, 5 * input_dim))
        self.w2 = nn.Parameter(torch.Tensor(5 * input_dim, output_dim))
        self.w1_grad = None
        self.w2_grad = None
        self.activate = nn.Sigmoid()

    def forward(self, input):
        if self.training:
            std, mean = torch.std_mean(torch.abs(self.w1))
            mask = torch.abs(self.w1) >= mean  # - self.mask * std
            w1 = self.w1 * mask

            std, mean = torch.std_mean(torch.abs(self.w2))
            mask = torch.abs(self.w2) >= mean  # - self.mask * std
            w2 = self.w2 * mask

            output = torch.matmul(input, w1)
            output = torch.matmul(output, w2)
            output = self.activate(output)

            return output
        else:
            output = torch.matmul(input, self.w1)
            output = torch.matmul(output, self.w2)
            output = self.activate(output)
            return output

    def one_matrix(self):
        if self.training:
            std, mean = torch.std_mean(torch.abs(self.w1))
            mask = torch.abs(self.w1) >= mean  # - self.mask * std
            w1 = self.w1 * mask

            std, mean = torch.std_mean(torch.abs(self.w2))
            mask = torch.abs(self.w2) >= mean  # - self.mask * std
            w2 = self.w2 * mask
            return torch.mm(w1, w2)
        else:
            return torch.mm(self.w1, self.w2)


    def update(self):
        # for n, p in self.named_parameters():
        self.w1_grad = self.w1.grad.clone().detach()
        self.w2_grad = self.w2.grad.clone().detach()


class sub_layers(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(sub_layers, self).__init__()

        self.layers = nn.ParameterList([nn.Parameter(torch.Tensor(input_dim, 5 * input_dim)),
                                        nn.Parameter(torch.Tensor(5 * input_dim, output_dim))])

    def forward(self, input):
        for layer in self.layers:
            input = torch.matmul(input, layer)
        return input

    def one_matrix(self):
        return torch.mm(self.layers[0], self.layers[1])


class lambda_(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(lambda_, self).__init__()

        self.layers = nn.ParameterList([nn.Parameter(torch.Tensor(input_dim, 5 * output_dim)),
                                        nn.Parameter(torch.Tensor(5 * output_dim, output_dim))])
        self.activate = nn.Sigmoid()

    def forward(self, input):
        for layer in self.layers:
            input = torch.matmul(input, layer)
        input = self.activate(input)
        return input

    def one_matrix(self):
        return torch.mm(self.layers[0], self.layers[1])
