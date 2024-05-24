import torch
import torch.cuda
from torch import nn
from modules.motion_estimator import MotionEstimator
from modules.camera_head import CameraHead
from modules.identity import Identity



class EgoMotionFilter(nn.Module):
    """
    Models the motion of the camera recording the scene (motion of the car)
    
    Given two consecutive frames, computes the camera transformation T from the camera coordinates of the first frame
    to the second.


    """
    def __init__(self) -> None:
        super().__init__()
        self.motion_estimator = MotionEstimator()
        #self.rnn = Identity() #nn.GRUCell(128, 128)
        self.camera_head = CameraHead()


    def forward(self, embedding1, embedding2, camera_state=None):
        batch_size = embedding1.shape[0]
        if camera_state is None:
            camera_state = torch.zeros(batch_size, 128, device=embedding1.device)

        x = self.motion_estimator(embedding1, embedding2)
        #x = self.rnn(x, camera_state)
        #camera_state = x.detach()
        translation, rotation_sinus = self.camera_head(x)
        roll, pitch, yaw = torch.asin(rotation_sinus).unbind(1)
        tensor_0 = torch.zeros(batch_size, device=embedding1.device)
        tensor_1 = torch.ones(batch_size, device=embedding1.device)

        RX = torch.stack([
                        torch.stack([tensor_1, tensor_0, tensor_0]),
                        torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                        torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).permute(2,0,1)

        RY = torch.stack([
                        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).permute(2,0,1)

        RZ = torch.stack([
                        torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                        torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).permute(2,0,1)

        R = torch.bmm(RZ, RY)
        R = torch.bmm(R, RX)

        P = torch.cat([R, torch.unsqueeze(translation, 2)], 2)
        P = torch.cat([P, torch.unsqueeze(torch.stack([tensor_0, tensor_0, tensor_0, tensor_1]).permute(1,0), 1)], 1)

        return P#, camera_state