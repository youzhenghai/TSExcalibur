# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from asteroid.engine import System

# import torch

class SystemInformed(System):
    def common_step(self, batch, batch_nb, train=True):

        inputs, targets, enrolls, aux_len, spk_id = batch

        est_targets, spk_pre = self(inputs, enrolls, aux_len)


        loss, sisdr_loss, ce_loss = self.loss_func(est_targets, targets, spk_pre, spk_id)
        return loss, sisdr_loss, ce_loss

    def training_step(self, batch, batch_nb):

        loss, sisdr_loss, ce_loss = self.common_step(batch, batch_nb, train=True)
        self.log("loss", loss, logger=True)
        self.log("sisdr_loss", sisdr_loss, logger=True)
        self.log("ce_loss", ce_loss, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        # print("validation_step")

        _, sisdr_loss, _ = self.common_step(batch, batch_nb, train=False)

        self.log("val_loss", sisdr_loss, on_epoch=True, prog_bar=True, sync_dist=True)


    # def train_dataloader(self):
    #     """Training dataloader"""
    #     # 初始化 DataLoader，这时会根据当前 rank 随机划分数据集
    #     self.train_loader.init_loader()
    #     # print("finish init train_loader")
    #     return self.train_loader

    # def val_dataloader(self):
    #     """Validation dataloader"""
    #     self.val_loader.init_loader()
    #     return self.val_loader