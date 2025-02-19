import os
import sys
import argparse
import json
import sys
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


sys.path.append('./../../../')
from calibur.model.dprnn_spe import DPRNNSpeTasNet
from calibur.datasets.datasets_dc import LibriMixInformed_dc
from calibur.model.system import SystemInformed
from asteroid.engine.optimizers import make_optimizer
from asteroid.losses import singlesrc_neg_sisdr

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")

def joint_loss(est_targets, targets, spk_pre, spk_id):

    sisdr_loss = singlesrc_neg_sisdr(est_targets[:,0], targets[:,0]).mean()
    ce = torch.nn.CrossEntropyLoss()
    ce_loss = ce(spk_pre, spk_id)
    loss = sisdr_loss + 0.5 * ce_loss
    return loss, sisdr_loss, ce_loss

def main(conf):

    train_set = LibriMixInformed_dc(
        csv_dir=conf["data"]["train_dir"],
        utt_scp_file=conf["data"]["tr_utt_scp_file"],
        noise_scp_file=conf["data"]["noise_tr_scp_file"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        segment_aux=conf["data"]["segment_aux"],
        spk_list=conf["data"]["spk_list"],
    )

    val_set = LibriMixInformed_dc(
        csv_dir=conf["data"]["valid_dir"],
        utt_scp_file=conf["data"]["cv_utt_scp_file"],
        noise_scp_file=conf["data"]["noise_cv_scp_file"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        segment_aux=conf["data"]["segment_aux"],
        spk_list=conf["data"]["spk_list"],
        
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )


    model = DPRNNSpeTasNet(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
    )
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=conf['training']['reduce_patience'])
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = joint_loss
    system = SystemInformed(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
        
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy='ddp',
        devices="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        # use_distributed_sampler=False
        # sync_batchnorm=True
        
    )
    if conf["training"]["last_checkpoint_path"] is not None:
        print(conf["training"]["last_checkpoint_path"])

        trainer.fit(system,ckpt_path=conf["training"]["last_checkpoint_path"])
    else:
        print('trainer from scratch')

        trainer.fit(system)


    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()

    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
