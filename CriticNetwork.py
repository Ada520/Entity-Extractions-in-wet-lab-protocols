import torch
import torch.nn as nn
class Critic_Network(nn.Module):

    def adjust(pred_y,ground_y,delta):
        return 0 if (pred_y==ground_y and delta<0) or (pred_y!=ground_y and delta>0) else 1


    def __init__(self,size):
        super(Critic_Network, self).__init__()
        self.l1=nn.Linear(size,1000)
        self.l2=nn.Linear(1000,800)
        self.l3=nn.Linear(800,1)
        self.non_linear=nn.LeakyReLU()
    
    def forward(self,input_hx_cx):
        return self.l3(self.non_linear(self.l2(self.non_linear(self.l1(input_hx_cx)))))
