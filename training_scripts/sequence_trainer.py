from tqdm import tqdm
import numpy as np
import torch
import util
from torch import nn

# todo: update these for the task at hand
def train_epoch(model, ego_motion, train_loader, optimizer, epoch, device, writer):
    """ Training a model for one epoch """
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    depth_criterion = nn.L1Loss()
    camera_criterion = nn.MSELoss()

    loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for j, sequence in progress_bar:
        feat_state = None
        camera_state = None
        last_embedding = None
        last_position = None
        position_loss = None
        for i in range(sequence[0].shape[1]):
            images = sequence[0][:,i,...].to(device)
            segmentations = sequence[1][:,i,...].to(device)
            depths = sequence[2][:,i,...].to(device)
            position = sequence[3][:,i,...].to(device)

            optimizer.zero_grad()

            pred_seg, pred_depth, feat_state, embedding = model(images, feat_state)
            
            seg_loss = seg_criterion(pred_seg, segmentations)
            depth_loss = depth_criterion(pred_depth, depths) / 6000
            loss_list.append(seg_loss.item() + depth_loss.item())
            loss=seg_loss+depth_loss
            loss.backward()

            # We only do the movement calculation if we have two frame of information
            if last_embedding is not None:
                # Model takes in two images and the initial position and returns a transition
                transform, camera_state = ego_motion(last_embedding, embedding, camera_state)

                position_loss = camera_criterion(torch.bmm(transform, last_position), position)
                position_loss.backward()
                
            optimizer.step()
            
            desc = f"Epoch {epoch+1}: seg_loss {seg_loss.item():.5f}, depth_loss {depth_loss.item():.5f}"
            if position_loss is not None:
                desc +=  f", pos_loss {position_loss.item():.5f}"
            progress_bar.set_description(desc)#f"Epoch {epoch+1}: seg_loss {seg_loss.item():.5f}, depth_loss {depth_loss.item():.5f}")

            last_embedding, last_position = embedding, position
            torch.cuda.empty_cache()

        if writer:
            iter_ = epoch * len(train_loader) + i
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'Depth_Loss/Train Iters', seg_loss.item(), global_step=iter_)
            writer.add_scalar(f'Seg Loss/Train Iters', depth_loss.item(), global_step=iter_)
            if position_loss is not None:
                writer.add_scalar(f'Position Loss/Train Iters', position_loss.item(), global_step=iter_)
            writer.add_scalar(f'_Params/Learning Rate', lr, global_step=iter_)

    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, ego_motion, eval_loader, device, epoch=None, writer=None):
    """ Evaluating the model for either validation or test """

    correct_pixels = 0
    total_pixels = 0
    ious = []
    loss_list = []
    
    progress_bar = tqdm(eval_loader, total=len(eval_loader))

    #weights = torch.as_tensor([0.23, 6.04, 1, 75.00, 6.03, 62.70, 0.29, 0.49, 0.54, 0.45, 4.33, 85.95,
    #                    0.30, 20.72, 1, 1, 69.55, 34.69, 7.879, 29.00, 1, 5.89], device=device)
    #seg_criterion = nn.CrossEntropyLoss(weights, ignore_index=255)
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    depth_criterion = nn.L1Loss()
    camera_criterion = nn.MSELoss()

    for sequence in progress_bar:
        feat_state = None
        camera_state = None
        last_embedding = None
        last_position = None
        position_loss = None
        for i in range(sequence[0].shape[1]):
            images = sequence[0][:,i,...].to(device)
            segmentations = sequence[1][:,i,...].to(device)
            depths = sequence[2][:,i,...].to(device)
            position = sequence[3][:,i,...].to(device)

            pred_seg, pred_depth, feat_state, embedding = model(images, feat_state)
            
            seg_loss = seg_criterion(pred_seg, segmentations)
            depth_loss = depth_criterion(pred_depth, depths) / 3000
            loss = seg_loss.item() + depth_loss.item()
            loss_list.append(loss)
            torch.cuda.empty_cache()

            # We only do the movement calculation if we have two frame of information
            if last_embedding is not None:
                # Model takes in two images and the initial position and returns a transition
                transform, camera_state = ego_motion(last_embedding, embedding, camera_state)
                position_loss = camera_criterion(torch.bmm(transform, last_position), position) * 5

            last_embedding, last_position = embedding, position
        
            desc = f"Epoch {epoch+1}: seg_loss {seg_loss.item():.5f}, depth_loss {depth_loss.item():.5f}"
            if position_loss is not None:
                desc +=  f", pos_loss {position_loss.item():.5f}"
            progress_bar.set_description(desc)#f"Eval Epoch {epoch+1} seg_loss {seg_loss.item():.5f}, depth_loss {depth_loss.item():.5f} ")
            # computing evaluation metrics
            predicted_class = torch.argmax(pred_seg, dim=1)
            correct_pixels += predicted_class.eq(segmentations).sum().item()
            total_pixels += segmentations.numel()
            iou = util.IoU(predicted_class, segmentations, num_classes=pred_seg.shape[1])
            ious.append(iou)
    
    # mean metrics and loss
    loss = np.mean(loss_list)
    avg_accuracy = 100 * correct_pixels / total_pixels   
    ious = torch.stack(ious)
    ious = ious.sum(dim=-1) / (ious != 0).sum(dim=-1)  # per class IoU
    mIoU = ious.mean()  # averaging across classes
    
    return loss, (avg_accuracy, mIoU)


def train_model(model, ego_motion, optimizer, scheduler, train_loader, valid_loader,
                num_epochs, writer, save_frequency=1, device="cpu", start_epoch=0):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    
    for epoch in range(start_epoch, num_epochs):
        log_epoch = (epoch % 5 == 0 or epoch == num_epochs - 1)
        
        # validation epoch
        model.eval()  # important for dropout and batch norms
        loss, (avg_accuracy, mIoU) = eval_model(
                model=model, ego_motion=ego_motion, eval_loader=valid_loader,
                device=device, epoch=epoch, writer=writer
            )
        
        val_loss.append(loss)
        writer.add_scalar(f'Loss/Valid', loss, global_step=epoch)
        writer.add_scalar(f'Metrics/Valid mAcc', avg_accuracy, global_step=epoch)
        writer.add_scalar(f'Metrics/Valid mIoU', mIoU, global_step=epoch)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        ego_motion.train()
        mean_loss, cur_loss_iters = train_epoch(
                model=model, ego_motion=ego_motion, train_loader=train_loader, optimizer=optimizer,
                 epoch=epoch, device=device, writer=writer
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
            #    "valid_loss": val_loss,
                "loss_iters": loss_iters
            }
            util.save_model(geometry_filter=model, ego_motion=ego_motion, optimizer=optimizer, epoch=epoch, stats=stats, savepath=f'models/finetune_ep_{epoch}.pth')
        
        if(log_epoch):
            print(f"    Train loss: {round(mean_loss, 5)}")
            #print(f"    Valid loss: {round(loss, 5)}")
    
    print(f"Training completed")
    return train_loss, val_loss, loss_iters


