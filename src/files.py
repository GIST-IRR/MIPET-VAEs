def make_run_files(args):
    if args.model_type == "mpl" or args.model_type == "vae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_vae".format(args.optimizer,
                                                                                args.num_epoch,
                                                                                args.lr_rate,
                                                                                args.seed,
                                                                                args.weight_decay,
                                                                                args.train_batch_size,
                                                                                args.latent_dim)
        return file

    elif args.model_type == "betavae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_beta:{}_dim:{}_beta".format(args.optimizer,
                                                                                         args.num_epoch,
                                                                                         args.lr_rate,
                                                                                         args.seed,
                                                                                         args.weight_decay,
                                                                                         args.train_batch_size,
                                                                                         args.beta,
                                                                                         args.latent_dim)
        return file
    elif args.model_type == "infovae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_alpha:{}_labmda:{}_dim:{}_info".format(args.optimizer,
                                                                                                    args.num_epoch,
                                                                                                    args.lr_rate,
                                                                                                    args.seed,
                                                                                                    args.weight_decay,
                                                                                                    args.train_batch_size,
                                                                                                    args.beta,
                                                                                                    args.lamb,
                                                                                                    args.latent_dim)
        return file

    elif args.model_type =="factorvae":
        file = "opt:{}_epoch:{}_lr:{}_dis_lr:{}_seed:{}_wd:{}_batch:{}_gamma:{}_dim:{}_factor".format(args.optimizer,
                                                                                                      args.num_epoch,
                                                                                                      args.lr_rate,
                                                                                                      args.discri_lr_rate,
                                                                                                      args.seed,
                                                                                                      args.weight_decay,
                                                                                                      args.train_batch_size,
                                                                                                      args.gamma,
                                                                                                      args.latent_dim)
        return file

    elif args.model_type == "betatcvae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_alpha:{}_beta:{}_labmda:{}_dim:{}_betatc".format(args.optimizer,
                                                                                                    args.num_epoch,
                                                                                                    args.lr_rate,
                                                                                                    args.seed,
                                                                                                    args.weight_decay,
                                                                                                    args.train_batch_size,
                                                                                                    args.alpha,
                                                                                                    args.beta,
                                                                                                    args.lamb,
                                                                                                    args.latent_dim)
        return file

    elif args.model_type =="mipetvae" or args.model_type == "mipetvae_reg":
        file = "opt:{}_epoch:{}_lr:{}_sub_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_inv-equ:{}_mipet".format(args.optimizer,
                                                                                                        args.num_epoch,
                                                                                                        args.lr_rate,
                                                                                                        args.sub_lr_rate,
                                                                                                        args.seed,
                                                                                                        args.weight_decay,
                                                                                                        args.train_batch_size,
                                                                                                        args.latent_dim,
                                                                                                      args.num_inv_equ)
        return file

    elif args.model_type =="commutativevae":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_hes:{}_commu:{}_rec:{}_commtative".format(args.optimizer,
                                                                                                args.num_epoch,
                                                                                                args.lr_rate,
                                                                                                args.seed,
                                                                                                args.weight_decay,
                                                                                                args.train_batch_size,
                                                                                                args.latent_dim,
                                                                                                args.hy_hes,
                                                                                                args.hy_commute,
                                                                                                args.hy_rec)
        return file

    elif args.model_type == "mipetbetatcvae" or args.model_type == "mipetbetatcvae_reg":
        file = "opt:{}_epoch:{}_lr:{}_sub_lr:{}_seed:{}_wd:{}_batch:{}_alpha:{}_beta:{}_labmda:{}_dim:{}_inv-equ:{}_mipetbetatc".format(args.optimizer,
                                                                                                                                    args.num_epoch,
                                                                                                                                    args.lr_rate,
                                                                                                                                    args.sub_lr_rate,
                                                                                                                                    args.seed,
                                                                                                                                    args.weight_decay,
                                                                                                                                    args.train_batch_size,
                                                                                                                                    args.alpha,
                                                                                                                                    args.beta,
                                                                                                                                    args.lamb,
                                                                                                                                    args.latent_dim,
                                                                                                                                    args.num_inv_equ)
        return file
    elif args.model_type == "mipetcommutativevae":
        file = "opt:{}_epoch:{}_lr:{}_sub_lr:{}_seed:{}_wd:{}_batch:{}_dim:{}_hes:{}_commu:{}_rec:{}_labmda:{}_inv-equ:{}_commtative".format(args.optimizer,
                                                                                                                                             args.num_epoch,
                                                                                                                                             args.lr_rate,
                                                                                                                                             args.sub_lr_rate,
                                                                                                                                             args.seed,
                                                                                                                                             args.weight_decay,
                                                                                                                                             args.train_batch_size,
                                                                                                                                             args.latent_dim,
                                                                                                                                             args.hy_hes,
                                                                                                                                             args.hy_commute,
                                                                                                                                             args.hy_rec,
                                                                                                                                             args.lamb,
                                                                                                                                             args.num_inv_equ)
        return file