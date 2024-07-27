import torch
import torch.nn.functional as F
import numpy as np

def compute_similarity_q(vals, mask, temp):
    pos_vals=torch.mul(vals,mask)
    neg_vals=torch.mul(vals,1-mask)







class ContrastiveLoss_fg_bg(torch.nn.Module):
  def __init__(self, temp, num_sim=5, num_dis_sim=15):
      super(ContrastiveLoss_fg_bg, self).__init__()
      self.temp=temp
      self.num_sim=num_sim
      self.num_dis_sim=num_dis_sim
  def forward(self, embedded_fg, embedded_bg, embedded_fg_2, embedded_bg_2):
      
      embedded_fg = F.normalize(embedded_fg, dim=1)
      embedded_fg_2 = F.normalize(embedded_fg_2, dim=1)
      embedded_bg = F.normalize(embedded_bg, dim=1)
      embedded_bg_2 = F.normalize(embedded_bg_2, dim=1)
      #sim_fg_fg = torch.div(torch.sum(torch.mul(embedded_fg, embedded_fg_2), dim=-1, keepdim=True), self.temp)
      sim_fg_bg2 = torch.div(torch.matmul(embedded_fg, embedded_bg_2.T), self.temp)
      sim_fg_bg2 = torch.topk(sim_fg_bg2,self.num_dis_sim, axis=-1).values
      sim_fg_fg = torch.matmul(embedded_fg, embedded_fg_2.T)
      mask_rm=torch.ones((embedded_fg.shape[0], self.num_sim)).cuda()
      mask_rm[:,0]=0
      sim_fg_fg = torch.topk(sim_fg_fg, self.num_sim, axis=-1, largest=True).values*mask_rm



      sim_fg_bg = torch.div(torch.matmul(embedded_fg, embedded_bg.T), self.temp)
      sim_fg_bg = torch.topk(sim_fg_bg, torch.minimum(torch.tensor(self.num_dis_sim),torch.tensor(embedded_bg.shape[0])), axis=-1).values

      #sim_fg_bg = torch.div(torch.sum(torch.mul(embedded_fg, embedded_bg), dim=-1, keepdim=True), self.temp)

      sim_bg_bg2 = torch.div(torch.matmul(embedded_bg, embedded_bg_2.T), self.temp)
      sim_bg_bg2 = torch.topk(sim_bg_bg2, self.num_sim, axis=-1, largest=True).values*mask_rm

      fg_fg_mask=torch.ones(size=sim_fg_fg.size()).cuda()
      bg_bg_mask=torch.ones(size=sim_bg_bg2.size()).cuda()
      fg_bg_mask=torch.zeros(size=sim_fg_bg.size()).cuda()
      fg_bg2_mask=torch.zeros(size=sim_fg_bg2.size()).cuda()
      total_sim = torch.cat([sim_fg_fg,sim_bg_bg2, sim_fg_bg2, sim_fg_bg], -1)
      #sim_labels = torch.zeros(embedded_fg.size(dim=0), dtype=torch.uint8).cuda()
      sim_labels=torch.cat([fg_fg_mask,bg_bg_mask, fg_bg2_mask, fg_bg_mask], -1)
      normalized_sim_labels=sim_labels/torch.sum(sim_labels, axis=-1, keepdim=True)

      loss = torch.nn.CrossEntropyLoss()(total_sim, normalized_sim_labels)
      return loss


def compute_sims(embedded_fg, embedded_bg, embedded_fg_2, embedded_bg_2):
    temp=.1
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_fg_2 = F.normalize(embedded_fg_2, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    embedded_bg_2 = F.normalize(embedded_bg_2, dim=1)
    #sim_fg_fg = torch.div(torch.sum(torch.mul(embedded_fg, embedded_fg_2), dim=-1, keepdim=True), temp)
    sim_fg_fg = torch.div(torch.matmul(embedded_fg, embedded_fg_2), temp)
    sim_fg_bg2 = torch.div(torch.matmul(embedded_fg, embedded_bg_2.T), temp)
    #sim_fg_fg = torch.matmul(embedded_fg, embedded_fg_2.T)
    sim_fg_bg = torch.div(torch.matmul(embedded_fg, embedded_bg.T), temp)
    total_sim = torch.cat([sim_fg_fg, sim_fg_bg, sim_fg_bg2], -1)
    sim_labels=torch.zeros(embedded_fg.size(dim=0))
    loss=torch.nn.CrossEntropyLoss()(total_sim, sim_labels)
