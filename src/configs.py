class VAEConfig():
    def __init__(self,
                 args,
                 hidden_states=True):
        self.dataset = args.dataset
        self.dense_dim = args.dense_dim
        self.latent_dim = args.latent_dim
        self.hidden_states = hidden_states
        self.num_sampling = args.num_sampling

class VanillaVAEConfig(VAEConfig):
    def __init__(self,
                 args,
                 input_size = 784):
        super(VanillaVAEConfig, self).__init__(args)
        self.input_size = input_size
        #self.hidden_size = args.hidden_size


class VanillaVAEConfig(VAEConfig):
    def __init__(self,
                 args,
                 in_channel=1):
        super(VanillaVAEConfig, self).__init__(args)
        self.in_channel = in_channel
        #self.hidden_size = args.hidden_size
        #self.latent_dim = args.latent_dim


class BetaVAEConfig(VAEConfig):
    def __init__(self,
                 args,
                 in_channel=1):
        super(BetaVAEConfig, self).__init__(args)
        self.in_channel = in_channel
        #self.hidden_size = args.hidden_size
        #self.latent_dim = args.latent_dim
        self.alpha = args.alpha
        self.beta = args.beta
        self.lamb = args.lamb

class InfoVAEConfig(VAEConfig):
    def __init__(self,
                 args,
                 in_channel=1):
        super(InfoVAEConfig, self).__init__(args)
        self.in_channel = in_channel
        #self.hidden_size = args.hidden_size
        #self.latent_dim = args.latent_dim
        self.alpha = args.beta
        self.lamb = args.lamb

class BetaTCVAEConfig(VAEConfig):
    def __init__(self,
                 args,
                 in_channel=1,
                 dataset_size=0):
        super(BetaTCVAEConfig, self).__init__(args)
        self.in_channel = in_channel
        self.dataset_size = dataset_size

class MIPET_Config(VAEConfig):
    def __init__(self,
                 args,
                 in_channel=1):
        super(MIPET_Config, self).__init__(args)
        self.batch_size = args.per_gpu_train_batch_size
        self._lambda = args.lamb
        self.bayesian = args.bayesian
        self.num_inv_equ = args.num_inv_equ

class LieConfig(VAEConfig):
    def __init__(self, args, in_channel=1):
        super(LieConfig, self).__init__(args)
        self.subspace_sizes_ls = args.subspace_sizes_ls # list of int
        self.subgroup_sizes_ls = args.subgroup_sizes_ls # list of int
        self.no_exp = args.no_exp
        self.hy_hes = args.hy_hes
        self.hy_rec =args.hy_rec
        self.hy_commute = args.hy_commute
        self.forward_eq_prob = args.forward_eq_prob

class MIPET_BetaTCConfig(BetaTCVAEConfig):
    def __init__(self, args, dataset_size):
        super(MIPET_BetaTCConfig, self).__init__(args, dataset_size=dataset_size)
        self.batch_size = args.per_gpu_train_batch_size
        self._lambda = args.lamb
        self.bayesian = args.bayesian
        self.num_inv_equ = args.num_inv_equ

class MIPET_LieConfig(LieConfig):
    def __init__(self, args, in_channel=1):
        super(MIPET_LieConfig, self).__init__(args)
        self.batch_size = args.per_gpu_train_batch_size
        self._lambda = args.lamb
        self.bayesian = args.bayesian
        self.num_inv_equ = args.num_inv_equ

