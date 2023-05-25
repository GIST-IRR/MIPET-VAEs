import os
import csv
import argparse

def write_info(args, results):
    info = None
    if args.model_type == "vae":
        info = VanillaInfo(args, **results)
    elif args.model_type == "betavae":
        info = BetaInfo(args, **results)
    elif args.model_type =="infovae":
        info = InfoInfo(args, **results)
    elif args.model_type == "betatcvae":
        info = BetaTCInfo(args, **results)
    elif args.model_type == "mipetvae" or args.model_type =="mipetvae_reg":
        info = MIPET_Info(args, **results)
    elif args.model_type == "commutativevae":
        info = CommutativeInfo(args, **results)
    elif args.model_type == "mipetbetatcvae" or args.model_type == "mipetbetatcvae_reg":
        info = MIPET_BetaTCInfo(args, **results)
    elif args.model_type == "mipetcommutativevae":
        info = MIPET_CommutativeInfo(args, **results)
    info.write_results()
    return

class VanillaInfo():

    def __init__(self, args, **kwargs):
        self.file_dir = args.results_file
        self.opt = args.optimizer
        self.epoch = args.num_epoch
        self.lr = args.lr_rate
        self.seed = args.seed
        self.wd = args.weight_decay
        self.batch = args.train_batch_size
        self.latent = args.latent_dim
        #self.alpha = args.alpha
        self.beta = args.beta
        self.lamb = args.lamb
        self.elbo = kwargs['elbo']
        self.obj = kwargs['obj']
        self.reconst = kwargs['reconst']
        self.kld = kwargs['kld']
        self.factor_disent = kwargs['factor_disent']
        self.mig = kwargs['mig']
        self.sap = kwargs['sap']
        self.dci_disent = kwargs['dci_disent']
        self.dci_completness = kwargs['dci_comple']
        #self.disen_acc = kwargs['disen_acc']
        #self.act_dim = kwargs['act_dims']

    def write_results(self):

        file_exists = os.path.isfile(self.file_dir)
        fieldnames = [str(key) for key in self.__dict__]

        with open(self.file_dir, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames= fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.__dict__)
        return

class BetaInfo(VanillaInfo):

    def __init__(self, args, **kwargs):
        super(BetaInfo, self).__init__(args, **kwargs)
        self.beta = args.beta

class InfoInfo(VanillaInfo):

    def __init__(self, args, **kwargs):
        super(InfoInfo, self).__init__(args, **kwargs)
        self.beta = args.beta
        self.lamb = args.lamb
        self.mmd = kwargs['mmd']

class FactorInfo(VanillaInfo):
    def __init__(self, args, **kwargs):
        super(FactorInfo, self).__init__(args, **kwargs)
        self.gamma = args.gamma
        self.dis_lr = args.discri_lr_rate
        self.tc = kwargs['tc']

class BetaTCInfo(VanillaInfo):

    def __init__(self, args, **kwargs):
        super(BetaTCInfo, self).__init__(args, **kwargs)
        self.alpha = args.alpha
        self.beta = args.beta
        self.lamb = args.lamb
        self.tc = kwargs['tc']
        self.mi = kwargs['mi']

class MIPET_Info(VanillaInfo):

    def __init__(self, args, **kwargs):
        super(MIPET_Info, self).__init__(args, **kwargs)
        #self.lamb = args.lamb
        self.sub_lr_rate = args.sub_lr_rate
        self.reg = kwargs['reg']
        self.num_inv_equ = args.num_inv_equ
        #self.prob = args.prob
class CommutativeInfo(VanillaInfo):

    def __init__(self, args, **kwargs):
        super(CommutativeInfo, self).__init__(args, **kwargs)
        self.hy_hes = args.hy_hes
        self.hy_rec = args.hy_rec
        self.hy_commute = args.hy_commute
        self.forward_eq_prob = args.forward_eq_prob
        self.group = kwargs['group']

class MIPET_BetaTCInfo(BetaTCInfo, MIPET_Info):
    def __init__(self,args, **kwargs):
        super(MIPET_BetaTCInfo, self).__init__(args, **kwargs)

class MIPET_CommutativeInfo(CommutativeInfo, MIPET_Info):
    def __init__(self, args, **kwargs):
        super(MIPET_CommutativeInfo, self).__init__(args, **kwargs)