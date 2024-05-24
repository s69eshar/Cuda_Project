from tqdm import tqdm
import numpy as np
import torch
import util

# todo: update these for the task at hand
def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """ Training a model for one epoch """
    
    resnet = util.get_pretrained_resnet("models/segmenter_epoch_8.pth").to(device)
    for param in resnet.parameters():
        param.requires_grad = False

    loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (image1, image2, p1, p2) in progress_bar:
        
        optimizer.zero_grad()
         
        image1, image2, p1, p2 = image1.to(device), image2.to(device), p1.to(device), p2.to(device)

        embedding1, embedding2 = resnet(image1)['out'], resnet(image2)['out']
        # Model takes in two images and the initial position and returns a transition
        transform = model(embedding1, embedding2)
         

        # TODO: fix the loss
        # The losses are based on the relative transformation between the predicted and ground-truth motion

        loss = criterion(torch.bmm(transform, p1), p2)
        loss_list.append(loss.item())
         
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
        progress_bar.set_description(f"Epoch {epoch+1}: loss {loss.item():.5f}. ")
        
    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device, epoch=None, writer=None):
    """ Evaluating the model for either validation or test """
    loss_list = []

    resnet = util.get_pretrained_resnet("models/segmenter_epoch_8.pth").to(device)
    for param in resnet.parameters():
        param.requires_grad = False
    progress_bar = tqdm(enumerate(eval_loader), total=len(eval_loader))
    for i, (image1, image2, p1, p2) in progress_bar:
        

        image1, image2, p1, p2 = image1.to(device), image2.to(device), p1.to(device), p2.to(device)

        embedding1, embedding2 = resnet(image1)['out'], resnet(image2)['out']
        # Model takes in two images and the initial position and returns a transition
        transform = model(embedding1, embedding2)
        

        loss = criterion(torch.bmm(transform, p2), p1)
        loss_list.append(loss.item())

        progress_bar.set_description(f"Eval Epoch {epoch+1} Loss = {loss.item():.5f}")    
    # Total correct predictions and loss
    loss = np.mean(loss_list)
    
    return loss


def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader,
                num_epochs, writer, save_frequency=2, device="cpu"):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    
    for epoch in range(num_epochs):
        log_epoch = (epoch % 5 == 0 or epoch == num_epochs - 1)
        # validation epoch
        model.eval()  # important for dropout and batch norms
        loss = eval_model(
                model=model, eval_loader=valid_loader, criterion=criterion,
                device=device, epoch=epoch, writer=writer
            )
        
        val_loss.append(loss)
        writer.add_scalar(f'Loss/Valid', loss, global_step=epoch)

        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device
            )
        writer.add_scalar(f'Loss/Train', mean_loss, global_step=epoch)
        #writer.add_scalars(f'Loss/Comb', {"train": mean_loss.item(), "valid": loss.item()}, global_step=epoch)
        
        # PLATEAU SCHEDULER
        scheduler.step(val_loss[-1])
        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters
        
        if(epoch % save_frequency == 0):
            stats = {
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "loss_iters": loss_iters
            }
            util.save_model(model=model, optimizer=optimizer, epoch=epoch, stats=stats, savepath=f"models/ego_motion_epoch_{epoch}.pth")
        
        if(log_epoch):
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid loss: {round(loss, 5)}")
    
    print(f"Training completed")
    return train_loss, val_loss, loss_iters


