import os
from sched import scheduler
import time

import torch
import wandb
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data

from dataset import custom_dataset
from loss import Loss
from model import EAST
from utils import get_lr
from eval import eval_torch_model


def train(
    train_img_path,
    train_gt_path,
    pths_path,
    batch_size,
    lr,
    num_workers,
    epoch_iter,
    save_interval,
):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[epoch_iter // 2], gamma=0.1
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=1200)
    for epoch in range(epoch_iter):
        model.train()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            # total step is current total step numbers from training start
            total_step = epoch * len(train_loader) + i
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = (
                img.to(device),
                gt_score.to(device),
                gt_geo.to(device),
                ignored_map.to(device),
            )
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
            
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            print(
                "Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}".format(
                    epoch + 1,
                    epoch_iter,
                    i + 1,
                    int(file_num / batch_size),
                    time.time() - start_time,
                    loss.item(),
                )
            )
            # upload the learning rate and loss after current step
            wandb.log(
                {"learning rate": get_lr(optimizer), "loss": loss.item()},
                step=total_step,
            )
            
        print(
            "epoch_loss is {:.8f}, epoch_time is {:.8f}".format(
                epoch_loss / int(file_num / batch_size), time.time() - epoch_time
            )
        )
        print(time.asctime(time.localtime(time.time())))
        print("=" * 100)
        # evaluate and save the model every `save_interval` epochs, skip first `skip_eval` epochs
        if ((epoch + 1) > skip_eval) and ((epoch + 1) % save_interval == 0):
            # get the eval results in float
            acc, recall, f1 = eval_torch_model(model, test_img_path, submit_path)
            # upload evaluation results to wandb
            wandb.log({"acc": acc, "recall": recall, "f1": f1}, step=total_step)
            # restore model state to train (otherwise the model params would not be updated)
            model.train()
            state_dict = (
                model.module.state_dict() if data_parallel else model.state_dict()
            )
            torch.save(
                state_dict,
                os.path.join(pths_path, "model_epoch_{}.pth".format(epoch + 1)),
            )


if __name__ == "__main__":
    train_img_path = os.path.abspath("../ICDAR_2015/train_img")
    train_gt_path = os.path.abspath("../ICDAR_2015/train_gt")
    pths_path = "./pths/11Aug_resnet50"
    batch_size = 24
    lr = 1e-3
    num_workers = 4
    epoch_iter = 1200
    save_interval = 5
    skip_eval = 100
    test_img_path = os.path.abspath("../ICDAR_2015/test_img")
    submit_path = "./submit"
    # init the wandb project
    wandb.init(project="EAST-resnet50", entity="chunfei-fyp")
    # register current experiment config
    wandb.config = {
        "pths_path": pths_path,
        "batch_size": batch_size,
        "lr": lr,
        "num_workers": num_workers,
        "epoch_iter": epoch_iter,
        "save_interval": save_interval,
    }
    train(
        train_img_path,
        train_gt_path,
        pths_path,
        batch_size,
        lr,
        num_workers,
        epoch_iter,
        save_interval,
    )
