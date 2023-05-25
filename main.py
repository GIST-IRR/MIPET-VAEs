import torch
import torch.nn as nn
import json
import shutil
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import os
from torch.optim import SGD, Adam
import argparse
# data set
from src.dataloaders import dstripeDataLoader, _3DshapeDataLoader
from src.dataloaders import _3DcarDataLoader
from src.optimizer import get_constant_schedule, get_linear_schedule_with_warmup
from src.info import write_info
from src.seed import set_seed
from src.files import make_run_files
from src.disentangle_metrics.disentagle_test import estimate_all_distenglement
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pdb
from src.configs import *
from models.mipet_vae import MIPET_VAE
from models.mipet_betatcvae import MIPET_BetaTCVAE
from models.mipet_commutativevae import MIPET_CommutativeVAE

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_HIDDEN_DIM = {
    'dsprites': [256, 128],
    'shapes3d': [256, 256],
    'car': [256, 256]
}

DATALOADER = {
    'dsprites': dstripeDataLoader,
    'shapes3d': _3DshapeDataLoader,
    'car': _3DcarDataLoader,
}
MODEL_CLASSES = {
    "mipetvae": (MIPET_Config, MIPET_VAE),
    "mipetbetatcvae": (MIPET_BetaTCConfig, MIPET_BetaTCVAE),
    "mipetcommutativevae": (MIPET_LieConfig, MIPET_CommutativeVAE),
}

OPTIMIZER = {
    'sgd': SGD,
    'adam': Adam,
}


def train(data_loader, train_dataset, test_dataset, num_epochs, model, loss_fn, args):
    optimizer = None
    set_seed(args)
    save_files = make_run_files(args)
    run_file = os.path.join(args.run_file, args.model_type, save_files)
    if args.write:
        if os.path.exists(run_file):
            shutil.rmtree(run_file)
            os.makedirs(run_file)
        else:
            os.makedirs(run_file)
        tb_writer = SummaryWriter(run_file)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  drop_last=False, pin_memory=True)
    global_step = 0
    learning_rate = args.lr_rate
    t_total = len(train_dataloader) * args.num_epoch
    if args.optimizer == 'sgd':
        optimizer = OPTIMIZER[args.optimizer](model.parameters(),
                                              lr=learning_rate,
                                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        if args.model_type == 'mipetvae' or args.model_type == 'mipetbetatcvae' or args.model_type == 'mipetcommutativevae':
            main_list, sub_list = [], []
            for name, param in model.named_parameters():
                if 'encoder' in name or 'decoder' in name:
                    main_list.append(param)
                else:
                    sub_list.append(param)
            optimizer = OPTIMIZER[args.optimizer]([{'params': main_list},
                                                   {'params': sub_list,
                                                    'lr': args.sub_lr_rate,
                                                    'weight_decay': 0.0}],
                                                  lr=learning_rate,
                                                  betas=(0.9, 0.999),
                                                  weight_decay=args.weight_decay)

        else:
            optimizer = OPTIMIZER[args.optimizer](model.parameters(),
                                                  lr=learning_rate,
                                                  betas=(0.9, 0.999),
                                                  weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total) if args.scheduler == 'linear' else get_constant_schedule(
        optimizer)


    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.output_dir, args.model_type, save_files, "optimizer.pt")) and \
            os.path.isfile(os.path.join(args.output_dir, args.model_type, save_files, "scheduler.pt")):

        optimizer.load_state_dict(
            torch.load(os.path.join(args.output_dir, args.model_type, save_files, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.output_dir, args.model_type, save_files, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        student, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train
    logger.info("***********Running dStripe dataset Training***********")
    logger.info(" Num examples = %d", len(train_dataset))
    logger.info(" Num Epochs = %d", args.num_epoch)
    logger.info(" Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        " Total train batch size = %d",
        args.train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("Total optimization steps = %d", t_total)

    tr_elbo, logging_elbo = 0.0, 0.0
    tr_reconst_err, logging_reconst_err = 0.0, 0.0
    tr_kld_err, logging_kld_err = 0.0, 0.0
    tr_mi, logging_mi = 0.0, 0.0
    tr_mmd, logging_mmd = 0.0, 0.0
    tr_tc, logging_tc = 0.0, 0.0
    tr_dis_tc, logging_dis_tc = 0.0, 0.0
    tr_reg, logging_reg = 0.0, 0.0
    tr_approxi_mi, logging_approxi_mi = 0.0, 0.0
    tr_total_loss, logging_total_loss = 0.0, 0.0
    tr_mipet_kld, logging_mipet_kld = 0.0, 0.0


    train_loss = []
    mmd, tc, mi, approxi_mi, reg, group, origin_kld, dis_tc = None, None, None, None, None, None, None, None
    total_loss = None
    tc_loss = None
    set_seed(args)
    # Check evaluate metric in 0 iteration
    # results = evaluate(args, test_dataset, model)

    # disent_results = disent_metric_vae(dataset, model=model, batch_size=args.train_batch_size // 2,
    #                                   num_train=args.num_disen)
    # for key, value in results.items():
    #    tb_writer.add_scalar(key, value, global_step)
    # for key, value in disent_results.items():
    #    tb_writer.add_scalar(key, value, global_step)
    iteration_per_epoch = len(train_dataloader)

    model.zero_grad()
    for epoch in range(num_epochs):
        iteration = tqdm(train_dataloader, desc="Iteration")

        for i, (data, class_label) in enumerate(iteration):

            model.train()
            data = data.to(device)
            outputs = model(data, loss_fn)

            reconst_err, kld_err = outputs[0]['obj']['reconst'], \
                                   outputs[0]['obj']['kld'],

            if args.model_type == "mipetvae":
                reg = outputs[0]['obj']['reg']
                origin_kld = outputs[0]['obj']['origin_kld']
                total_loss = reconst_err + kld_err + reg
            elif args.model_type == "mipetbetatcvae":
                tc = outputs[0]['obj']['tc']
                mi = outputs[0]['obj']['mi']
                reg = outputs[0]['obj']['reg']
                origin_kld = outputs[0]['obj']['origin_kld']
                total_loss = reconst_err + args.alpha * mi + args.beta * tc + args.gamma * kld_err + reg
            elif args.model_type == "mipetcommutativevae":
                group = outputs[0]['obj']['group']
                reg = outputs[0]['obj']['reg']
                origin_kld = outputs[0]['obj']['origin_kld']
                total_loss = reconst_err + kld_err + group + reg

            if args.n_gpu > 1:
                total_loss = total_loss.mean()
                reconst_err = reconst_err.mean()
                kld_err = kld_err.mean()
                # zid_mean = zid_mean.mean()
                # zid_var = zid_var.mean()
                if args.model_type == "infovae":
                    mmd = mmd.mean()
                elif "factor" in args.model_type:  # args.model_type == "factorvae" or args.model_type == "mipetfactorvae":
                    tc = tc.mean()
                    tc_loss = tc_loss.mean()
                elif "betatc" in args.model_type:  # args.model_type == "betatcvae" or args.model_type == "mipetbetatcvae":
                    tc = tc.mean()
                    mi = mi.mean()
                elif "mipet" in args.model_type:  # args.model_type == "chicvae" or args.model_type == "chicfactorvae" or args.model_type == "chicbetatcvae":
                    reg = reg.mean()
                    origin_kld = origin_kld.mean()
                elif "commutative" in args.model_type:  # == "commutativevae":
                    group = group.mean()

            elbo = -(reconst_err + kld_err)
            tr_total_loss += total_loss.item()
            tr_elbo += elbo.item()
            tr_reconst_err += reconst_err.item()
            tr_kld_err += origin_kld.item() if "mipet" in args.model_type else kld_err.item()  # args.model_type != 'chicvae' else origin_kld.item()

            # tr_zid_mean += zid_mean.item()
            # tr_zid_var += zid_var.item()
            if args.model_type == "infovae":
                tr_mmd += mmd.item()
            elif "factor" in args.model_type:  # args.model_type == "factorvae":
                tr_tc += tc.item()
                tr_dis_tc += tc_loss.item()
            elif "betatc" in args.model_type:  # args.model_type == "betatcvae":
                tr_tc += tc.item()
                tr_mi += mi.item()
            elif "mipet" in args.model_type:  # args.model_type == "chicvae":
                tr_reg += reg.item()
                tr_mipet_kld += kld_err.item()
            # elif args.model_type == "ccasvae":
            #    tr_approxi_mi += approxi_mi.item()

            if args.fp16:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # pdb.set_trace()
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % (args.logging_steps) == 0:
                logs = {}

                logs["00.ELBO"] = (tr_elbo - logging_elbo) / args.logging_steps
                logs["01.Total_Loss"] = (tr_total_loss - logging_total_loss) / args.logging_steps
                logs["02.Reconstruction_Loss"] = (tr_reconst_err - logging_reconst_err) / args.logging_steps
                logs["03.Kullback-Reibler_Loss"] = (tr_kld_err - logging_kld_err) / args.logging_steps

                # logs["05.Posterior_ID"] = (tr_zid_mean - logging_zid_mean) / args.logging_steps
                # logs["06.Posterior_ID_Var"] = (tr_zid_var - logging_zid_var) / args.logging_steps
                if mmd != None:
                    logs["07.MMD"] = (tr_mmd - logging_mmd) / args.logging_steps
                if tc != None:
                    logs["04.Mutual_Information"] = - (tr_mi - logging_mi) / args.logging_steps
                    logs["08.TC"] = (tr_tc - logging_tc) / args.logging_steps
                if approxi_mi != None:
                    logs["09.Approximated MI"] = (tr_approxi_mi - logging_approxi_mi) / args.logging_steps
                if reg != None:
                    logs["10.Regularizer"] = (tr_reg - logging_reg) / args.logging_steps
                    logs["11.MIPET-kld_loss"] = (tr_mipet_kld - logging_mipet_kld) / args.logging_steps
                logging_elbo = tr_elbo
                logging_total_loss = tr_total_loss
                logging_reconst_err = tr_reconst_err
                logging_kld_err = tr_kld_err
                logging_mi = tr_mi
                # logging_zid_mean = tr_zid_mean
                # logging_zid_var = tr_zid_var
                if mmd != None:
                    logging_mmd = tr_mmd
                if tc != None:
                    logging_tc = tr_tc
                if approxi_mi != None:
                    logging_approxi_mi = tr_approxi_mi
                if reg != None:
                    logging_reg = tr_reg
                    logging_mipet_kld = tr_mipet_kld
                learning_rate_scalar = scheduler.get_last_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                for key, value in logs.items():
                    tb_writer.add_scalar(f'Train/{key}', value, global_step)
                # print(json.dumps({**logs, **{"step": global_step}}))

            if args.evaluate_during_training and global_step % (iteration_per_epoch) == 0 and args.split != 0.0:
                results = evaluate(args, test_dataset, model, loss_fn)
                # disent_results = FactorVAEMetric(data_loader, model=model, batch_size=args.batch_disen,
                #                                 num_train=args.num_disen)
                train_loss.append(results['elbo'])
                # early_stop.update(train_loss)
                for key, value in results.items():
                    tb_writer.add_scalar(f'Eval/{key}', value, global_step)
                # for key, value in disent_results.items():
                #    tb_writer.add_scalar(f'Eval/{key}', value, global_step)

            if (args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0) or \
                    global_step == iteration_per_epoch * args.num_epoch:  # save in last step
                output_dir = os.path.join(args.output_dir, args.model_type, save_files,
                                          "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (model.module if hasattr(model, "module") else model)
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            # if args.max_steps > 0 and global_step > args.max_steps:
            # iteration.close()
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return


# 여기서 부터 다시 작성할 것.
def evaluate(args, test_dataset, model, loss_fn):
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, drop_last=False, pin_memory=True)
    total_size = len(test_dataloader)  # * args.test_batch_size

    result = {'elbo': {}, 'obj': {}}
    save_files = make_run_files(args)
    checkpoint_dir = os.path.join(args.output_dir, args.model_type, save_files, 'imgs')
    images_list, rimages_list = [], []

    logger.info("***********Running dStripe dataset Evaluation***********")
    logger.info(" Num examples = %d", len(test_dataset))

    model.eval()
    with torch.no_grad():
        elbo, obj, zid_mean, zid_var = 0.0, 0.0, 0.0, 0.0
        reconst, kld, mi, mmd, tc, approxi_mi, reg, group = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        iteration = tqdm(test_dataloader, desc='Iteration')
        for _, (data, class_label) in enumerate(iteration):
            data = data.to(device)
            outputs = model(data, loss_fn)
            _reconst, _kld, = outputs[0]['obj']['reconst'], \
                              outputs[0]['obj']['kld']

            if args.n_gpu > 1:
                # total_loss = total_loss.mean()
                _reconst = _reconst.sum()
                _kld = _kld.sum()

            elbo += (_reconst + _kld).item()
            reconst += _reconst.item()
            kld += _kld.item()


            if args.model_type == "mipetbetatcvae":
                _tc = outputs[0]['obj']['tc'].sum() if args.n_gpu > 1 else outputs[0]['obj']['tc']
                _mi = outputs[0]['obj']['mi'].sum() if args.n_gpu > 1 else outputs[0]['obj']['mi']
                obj += (_reconst + args.alpha * _mi + args.beta * _tc + args.gamma * _kld).item()
                tc += _tc.item()
            elif args.model_type == "mipetvae":
                _reg = outputs[0]['obj']['reg'].sum() if args.n_gpu > 1 else outputs[0]['obj']['reg']
                obj += (_reconst + _kld - _reg).item()
                reg += _reg.item()
            elif args.model_type == "commutativevae":
                _group = outputs[0]['obj']['group'].sum() if args.n_gpu > 1 else outputs[0]['obj']['group']
                obj += (_reconst + _kld + _group).item()
                group += _group.item()
            elif args.model_type == "mipetcommutativevae":
                _group = outputs[0]['obj']['group'].sum() if args.n_gpu > 1 else outputs[0]['obj']['group']
                _reg = outputs[0]['obj']['reg'].sum() if args.n_gpu > 1 else outputs[0]['obj']['reg']
                obj += (_reconst + _kld + _group + _reg).item()
                group += _group.item()
                reg += _reg.item()

            reconsted_images = outputs[2][0] if "commutative" not in args.model_type else outputs[2]

        result['elbo'] = - elbo / total_size
        result['reconst'] = reconst / total_size
        result['kld'] = kld / total_size

        if args.model_type == "infovae":
            result['mmd'] = mmd / total_size
        if "factor" in args.model_type:  # args.model_type == "factorvae":
            result['tc'] = tc / total_size
        if "betatc" in args.model_type:  # args.model_type == "betatcvae":
            result['mi'] = - mi / total_size
            result['tc'] = tc / total_size
        if "mipet" in args.model_type:  # args.model_type =="chicvae":
            result['reg'] = reg / total_size
        if "commutative" in args.model_type:  # == "commutativevae":
            result['group'] = group / total_size
        result['obj'] = obj / total_size

    return result


def main():
    parser = argparse.ArgumentParser()
    # set device info
    parser.add_argument(
        "--device_idx",
        type=str,
        default='cuda:0',
        required=True,
        help='set GPU index, i.e. cuda:0,1,2 ...',
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=0,
        required=False,
        help='number of available gpu',
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # DATASETS
    parser.add_argument(
        "--data_dir",
        type=str,
        default='{dataset directory}',
        required=False,
        help='dataset directory',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['dsprites', 'shapes3d', 'car'],
        required=True,
        help='Choose Dataset',
    )

    # model save directory
    parser.add_argument(
        "--output_dir",
        type=str,
        default="{model checkpoint directory}",
        required=True,
        help="model saving directory"
    )
    parser.add_argument(
        "--run_file",
        type=str,
        default="{tensorboard runfile directory}",
        required=False,
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="{it will be set automatically}",
        required=False,
        help="model performance saving directory"
    )
    # set model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['mipetvae', 'mipetbetatcvae', 'mipetcommutativevae'],
        required=True,
        help='choose vae type'
    )
    parser.add_argument(
        "--dense_dim",
        nargs='*',
        default=[256, 128],
        type=int,
        required=False,
        help='set CNN hidden FC layers',
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=8,
        required=False,
        help='set prior dimension z',
    )
    # for model hyper-parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        required=False,
        help="Set hyper-parameter alpha"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        required=False,
        help="Set hyper-parameter beta"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        required=False,
        help="Set hyper-parameter gamma"
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default="1.0",
        required=False,
        help="Set hyper-parameter lambda"
    )
    # Factor VAE hyper-parameters
    parser.add_argument(
        "--discri_lr_rate",
        type=float,
        default=1e-4,
        required=False,
        help="Set discriminator learning rate"
    )
    # MIPET-VAE
    parser.add_argument(
        "--num_inv_equ",
        type=int,
        default=3,
        required=False,
        help="Number of MIPET-VAE invertible and equivariant function"
    )
    parser.add_argument(
        "--sub_lr_rate",
        default=1e-10,
        type=float,
        required=False,
        help="Set learning rate"
    )
    parser.add_argument(
        "--prob",
        default=0.5,
        type=float,
        required=False,
        help="Set probability for MIPET kl-divergence"
    )
    parser.add_argument(
        "--bayesian",
        action='store_true',
        help='Bayesian update for posterior',
    )
    # Commutative-VAE
    parser.add_argument(
        "--hy_hes",
        type=float,
        default=40.0,
        required=False,
        help="Set hyper-parameter for commutative-VAE"
    )
    parser.add_argument(
        "--hy_rec",
        type=float,
        default=0.1,
        required=False,
        help="Set hyper-parameter for commutative-VAE"
    )
    parser.add_argument(
        "--hy_commute",
        type=float,
        default=20.0,
        required=False,
        help="Set hyper-parameter for commutative-VAE"
    )
    parser.add_argument(
        "--forward_eq_prob",
        type=float,
        default=0.2,
        required=False,
        help="Set hyper-parameter for commutative-VAE"
    )
    parser.add_argument(
        "--subgroup_sizes_ls",
        nargs='*',
        default=[100],
        type=int,
        required=False,
        help="Set hyper-parameter for commutative-VAE"
    )
    parser.add_argument(
        "--subspace_sizes_ls",
        nargs='*',
        default=[10],
        type=int,
        required=False,
        help="Set hyper-parmeter for commutative-VAE"
    )
    parser.add_argument(
        "--no_exp",
        action='store_true',
    )
    parser.add_argument(
        "--num_sampling",
        type=int,
        default=1,
        required=False,
        help="Set hyper-parameter for samplings on TC-Beta-VAE"
    )
    parser.add_argument(
        "--lr_rate",
        default=1e-4,
        type=float,
        required=False,
        help="Set learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        required=False,
        help="Set weight decay"
    )
    # set training info
    parser.add_argument(
        "--split",
        type=float,
        default=0.0,
        required=False,
        help='set split ratio for train set and test set'
    )
    parser.add_argument(
        "--shuffle",
        action='store_true',
        help='whether shuffling dataset or not'
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of training mini-batch size"
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of training mini-batch size for multi GPU training"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of evaluation mini-batch size"
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=60,
        required=False,
        help="Set number of epoch size"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        required=False,
        help="Save model checkpoint iteration interval"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        required=False,
        help="Update tb_writer iteration interval"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        required=False,
        help="interval for early stopping"
    )
    parser.add_argument(
        "--optimizer",
        choices=['sgd', 'adam'],
        default='sgd',
        type=str,
        help="Choose optimizer",
        required=False,
    )
    parser.add_argument(
        "--scheduler",
        choices=['const', 'linear'],
        default='const',
        type=str,
        help="Whether using scheduler during training or not",
        required=False,
    )

    # set for disentanglement learning
    parser.add_argument(
        "--num_disen_train",
        type=int,
        default=10,
        required=False,
        help='set number of disentanglement evaluation task'
    )
    parser.add_argument(
        "--num_disen_test",
        type=int,
        default=10,
        required=False,
        help='set number of disentanglement evaluation task'
    )
    parser.add_argument(
        "--batch_disen",
        type=int,
        default=100,
        required=False,
        help='set batch for Factor VAE disentanglement learning'
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--do_train", action='store_true', help="Do training")
    parser.add_argument("--do_eval", action='store_true', help="Do evaluation")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--save_imgs", action="store_true", help="Do save imgs")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Whether write tensorboard or not",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help='set seed',
    )
    args = parser.parse_args()
    # set VAE dense dim by dataset.
    args.dense_dim = DATA_HIDDEN_DIM[args.dataset]
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # pdb.set_trace()

    #path = args.data_dir
    #shuffle_dataset = True
    #random_seed = 100

    data_loader = DATALOADER[args.dataset](path=args.data_dir,
                                           shuffle_dataset=True,
                                           random_seed=args.seed,
                                           split_ratio=args.split)  # datasets[args.dataset](args)
    train_datasets, valid_datasets = data_loader()
    # train_datasets, valid_datasets = data_loader.split()

    dataset_size = len(train_datasets)
    config, model = MODEL_CLASSES[args.model_type]
    config = config(args=args, dataset_size=dataset_size) if 'betatcvae' in args.model_type else config(args=args)
    model = model(config=config)
    model.init_weights()
    model.to(device)
    loss_fn = nn.BCELoss(reduction='sum') if args.dataset == 'dsprites' else nn.MSELoss(reduction='sum')
    if args.do_train and args.do_eval:
        train(data_loader=data_loader,
              train_dataset=train_datasets,
              test_dataset=train_datasets if args.split == 0.0 else valid_datasets,
              num_epochs=args.num_epoch,
              model=model,
              loss_fn=loss_fn,
              args=args)

        results = evaluate(args, valid_datasets, model, loss_fn) if args.split != 0.0 else evaluate(args,
                                                                                                    train_datasets,
                                                                                                    model, loss_fn)
        disent_results = estimate_all_distenglement(data_loader, model,
                                                    disent_batch_size=args.batch_disen,
                                                    disent_num_train=args.num_disen_train,
                                                    disent_num_test=args.num_disen_test,
                                                    loss_fn=loss_fn,
                                                    continuous_factors=False)
        results['factor_disent'] = disent_results['factor']['disentanglement_accuracy']
        results['mig'] = disent_results['mig']
        results['sap'] = disent_results['sap']
        results['dci_disent'] = disent_results['dci']['disent']
        results['dci_comple'] = disent_results['dci']['comple']

        save_files = make_run_files(args)
        output_dir = os.path.join(args.output_dir, args.model_type)  # , save_files)
        args.results_file = os.path.join(output_dir, "result.csv")
        write_info(args, results)

    elif args.do_eval:
        # disent_results = FactorVAEMetric(data_loader, model=model, batch_size=args.batch_disen, num_train=args.num_disen)
        disent_results = estimate_all_distenglement(data_loader, model,
                                                    disent_batch_size=args.batch_disen,
                                                    disent_num_train=args.num_disen_train,
                                                    disent_num_test=args.num_disen_test,
                                                    loss_fn=loss_fn,
                                                    continuous_factors=False)
        results = {}
        results['factor_disent'] = disent_results['factor']['disentanglement_accuracy']
        results['mig'] = disent_results['mig']
        results['sap'] = disent_results['sap']
        results['dci_disent'] = disent_results['dci']['disent']
        results['dci_comple'] = disent_results['dci']['comple']


if __name__ == "__main__":
    main()