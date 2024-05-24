from tqdm import tqdm
import numpy as np
import torch
import util


# todo: update these for the task at hand
def train_epoch(model, train_loader, optimizer, criterion, epoch, device, writer):
    """ Training a model for one epoch """
    
    loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, segmentations) in progress_bar:
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        images, segmentations = images.to(device), segmentations.to(device)

        # Forward pass
        result = model(images)
        predictions = result
         
        # Calculate Loss
        loss = criterion(predictions, segmentations)
        loss_list.append(loss.item())
         
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()

        if i % 30 == 0:
            iter_ = epoch * len(train_loader) + i
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'Loss/Train Iters', loss.item(), global_step=iter_)
            writer.add_scalar(f'_Params/Learning Rate', lr, global_step=iter_)
        
        progress_bar.set_description(f"Epoch {epoch+1}: loss {loss.item():.5f}. ")
        
    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device, epoch, writer):
    """ Evaluating the model for either validation or test """
    correct_pixels = 0
    total_pixels = 0
    ious = []
    loss_list = []
    for images, segmentations in tqdm(eval_loader):
        images, segmentations = images.to(device), segmentations.to(device)
        
        # Forward pass only to get logits/output
        outputs = model(images)    
        loss = criterion(outputs, segmentations)
        loss_list.append(loss.item())
            
        # computing evaluation metrics
        predicted_class = torch.argmax(outputs, dim=1)
        correct_pixels += predicted_class.eq(segmentations).sum().item()
        total_pixels += segmentations.numel()
        iou = util.IoU(predicted_class, segmentations, num_classes=outputs.shape[1])
        ious.append(iou)
    
    # mean metrics and loss
    loss = np.mean(loss_list)
    avg_accuracy = 100 * correct_pixels / total_pixels   
    ious = torch.stack(ious)
    ious = ious.sum(dim=-1) / (ious != 0).sum(dim=-1)  # per class IoU
    mIoU = ious.mean()  # averaging across classes

    return loss, (avg_accuracy, mIoU)


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
        loss, (avg_accuracy, mIoU) = eval_model(
                model=model, eval_loader=valid_loader, criterion=criterion,
                device=device, epoch=epoch, writer=writer
            )
        
        val_loss.append(loss)
        writer.add_scalar(f'Loss/Valid', loss, global_step=epoch)
        writer.add_scalar(f'Metrics/Valid mAcc', avg_accuracy, global_step=epoch)
        writer.add_scalar(f'Metrics/Valid mIoU', mIoU, global_step=epoch)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device, writer=writer
            )
        writer.add_scalar(f'Loss/Train', mean_loss, global_step=epoch)
        writer.add_scalars(f'Loss/Comb', {"train": mean_loss.item(), "valid": loss.item()}, global_step=epoch)
        
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
            util.save_segmenter(model=model, optimizer=optimizer, epoch=epoch, stats=stats, savepath=f"models/seg2_epoch_{epoch}.pth")
        
        if(log_epoch):
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid loss: {round(loss, 5)}")
    
    print(f"Training completed")
    return train_loss, val_loss, loss_iters


