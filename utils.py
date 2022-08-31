from dataset.BDD import BDD
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


def get_loader(images_dir: str,
               masks_dir: str,
               obj_cls: list,
               img_size: int = 600,
               stage: str = 'train',
               batch_size: int = 1,
               shuffle: bool = True, ):
    bdd = BDD(images_dir, masks_dir, obj_cls, img_size, stage)

    loader = DataLoader(dataset=bdd,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=bdd.collate_fn)

    return loader


def train_one_epoch(loader, model, optimizer, device):
    model.train()
    training_loss = []
    loop = tqdm(loader)
    for batch in loop:
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loop.set_postfix(loss=losses.item())
        training_loss += losses.item()
        losses.backward()
        optimizer.step()

    training_loss /= len(loader)

    return training_loss


def val_one_epoch(loader, model, device):
    model.eval()
    val_loss = []
    loop = tqdm(loader)
    for batch in loop:
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loop.set_postfix(loss=losses.item())
        val_loss += losses.item

    val_loss /= len(loader)

    return val_loss


def logger(training_loss, val_loss, epoch, path_to_file):
    log_string = f"Training Loss: {training_loss}, Val Loss: {val_loss}, Epoch: {epoch}"
    with open(path_to_file, 'a') as file:
        file.write(log_string)


def load_check_point(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    best_val_loss = checkpoint['best_val_loss']
    return model, optimizer, epoch, training_losses, validation_losses, best_val_loss


def save_checkpoint(epoch, model, optimizer, training_losses, validation_losses, best_val_loss, path_to_save):
    print("########################################")
    print("Saving model...")
    print("########################################")
    torch.save({
        'epoch': epoch,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path_to_save)
    print("########################################")
    print("Model saved")
    print("########################################")


def predict(model, image, device):
    pass


def export_losses(training_losses, validation_losses, path):
    data = {
        "Training Losses": training_losses,
        "validation Losses": validation_losses
    }

    df = pd.DataFrame(data=data)
    df.to_csv(path)
