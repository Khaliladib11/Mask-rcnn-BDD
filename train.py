import os
import torch
import torch.optim as optim
import argparse
from utils import *
from model.Maskrcnn import get_model

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

classes = ['__bgr__', 'person', 'car', 'rider', 'bicycle', 'motorcycle', 'truck', 'bus']

images_dir = "D:/City University of London/MSc Artificial intelligence/Term 3/Project/dataset/bdd100k/images/10k/bdd100k/images/10k"
masks_dir = "D:/City University of London/MSc Artificial intelligence/Term 3/Project/dataset/bdd100k/labels/ins_seg"



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
        model, optimizer, last_epoch, training_losses, validation_losses, best_val_loss = load_check_point(**checkpoint_params)
    else:
        training_losses = []
        validation_losses = []
        best_val_loss = 1000
        last_epoch = 0

    model.to(device)
    for epoch in range(last_epoch, total_epochs):

        train_loss = train_one_epoch(training_loader, model, optimizer, device)
        val_loss = val_one_epoch(validation_loader, model, device)

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
    parser.add_argument('--checkpoint', type=str, default=None, help='train from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--total_epochs', type=int, default=100, help='total_epochs')

    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    checkpoint = args.checkpoint
    img_size = args.img_size
    total_epochs = args.total_epochs

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
    training_losses, validation_losses = train(**train_params)

    export_losses(training_losses, validation_losses, os.path.join(path_to_save, 'losses.csv'))
