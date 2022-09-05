import os
import yaml
import torch
import torch.optim as optim
import argparse
from utils import *
from model.Maskrcnn import get_model

import wandb

wandb.init(project='MaskRCNN-BDD')


def train(training_loader,
          validation_loader,
          model,
          optimizer,
          device,
          total_epochs,
          path_to_save,
          checkpoint=None):

    print("########################################")
    print("Start Training")
    print("########################################")
    logger_path = os.path.join(path_to_save, 'results.txt')
    best_path = os.path.join(path_to_save, 'best.pt')
    last_path = os.path.join(path_to_save, 'last.pt')

    if checkpoint is not None:
        checkpoint_params = {
            'model': model,
            'optimizer': optimizer,
            'path': checkpoint
        }
        model, optimizer, last_epoch, training_losses, validation_losses, best_val_loss = load_check_point(
            **checkpoint_params)
    else:
        training_losses = []
        validation_losses = []
        best_val_loss = 1000
        last_epoch = 0

    wandb.watch(model, log="all", log_freq=10)

    model.to(device)
    for epoch in range(last_epoch, total_epochs):

        train_loss = train_one_epoch(training_loader, model, optimizer, device)
        val_loss = val_one_epoch(validation_loader, model, device)

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        if val_loss < best_val_loss:
            save_checkpoint(epoch, model, optimizer, train_loss, val_loss, best_val_loss, best_path)

        logger(train_loss, val_loss, epoch, logger_path)

        save_checkpoint(epoch, model, optimizer, training_losses, validation_losses, best_val_loss, last_path)

    return training_losses, validation_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--data', type=str, default="data/data.yaml", help='data.yaml path')
    parser.add_argument('--checkpoint', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--total_epochs', type=int, default=100, help='total_epochs')

    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    checkpoint = args.checkpoint
    img_size = args.img_size
    total_epochs = args.total_epochs
    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)

    images_dir = data['images']
    masks_dir = data['masks']
    classes = data['classes']

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print(f"We are using {device}")

    train_loader_params = {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "obj_cls": classes,
        "img_size": img_size,
        "stage": 'train',
        "batch_size": batch_size,
        "shuffle": True,
    }

    train_loader = get_loader(**train_loader_params)

    val_loader_params = {
        "images_dir": images_dir,
        "masks_dir": masks_dir,
        "obj_cls": classes,
        "img_size": img_size,
        "stage": 'test',
        "batch_size": batch_size,
        "shuffle": True,
    }

    val_loader = get_loader(**val_loader_params)

    model = get_model(num_classes=len(classes))

    opt_pars = {'lr': lr, 'weight_decay': 1e-3}
    optimizer = optim.Adam(list(model.parameters()), **opt_pars)
    path_to_save = "./runs/train"

    train_params = {
        "training_loader": train_loader,
        "validation_loader": val_loader,
        "model": model,
        "optimizer": optimizer,
        "device": device,
        "total_epochs": total_epochs,
        "path_to_save": path_to_save,
        "checkpoint": checkpoint
    }

    wandb.config = {
        "learning_rate": lr,
        "epochs": total_epochs,
        "batch_size": batch_size
    }

    training_losses, validation_losses = train(**train_params)

    export_losses(training_losses, validation_losses, os.path.join(path_to_save, 'losses.csv'))
