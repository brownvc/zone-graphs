import sys
sys.path.append('..')

import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch.autograd import Variable

class GraphNetSoftMax(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphNetSoftMax, self).__init__()

        self.node_feat_size = in_dim

        self.mp1 = MPLayer(self.node_feat_size, self.node_feat_size)
        self.post_mp1 = nn.Sequential(
                            nn.BatchNorm1d(self.node_feat_size),
                            nn.LeakyReLU(negative_slope=0.2))

        self.mp2 = MPLayer(self.node_feat_size, self.node_feat_size)
        self.post_mp2 = nn.Sequential(
                            nn.BatchNorm1d(self.node_feat_size),
                            nn.LeakyReLU(negative_slope=0.2))
        
        self.mp3 = MPLayer(self.node_feat_size, self.node_feat_size)
        self.post_mp3 = nn.Sequential(
                            nn.BatchNorm1d(self.node_feat_size),
                            nn.LeakyReLU(negative_slope=0.2))

        self.mp4 = MPLayer(self.node_feat_size, self.node_feat_size)
        self.post_mp4 = nn.Sequential(
                            nn.BatchNorm1d(self.node_feat_size),
                            nn.LeakyReLU(negative_slope=0.2))

        self.fc1 = nn.Sequential(nn.Linear(self.node_feat_size, 128),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(negative_slope=0.2))

        self.fc2 = nn.Sequential(nn.Linear(128, 128),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(negative_slope=0.2))

        self.fc_final = nn.Sequential(nn.Linear(128, out_dim),
                            nn.LogSoftmax(dim=1))
    
        
    def forward(self, g):
        h = g.ndata.pop('h')
        
        h = self.mp1(g, h)
        h = self.post_mp1(h)

        h = self.mp2(g, h)
        h = self.post_mp2(h)

        h = self.mp3(g, h)

        h = self.readout(g, h)

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc_final(h)
   
        return h

    def readout(self, g, node_feats):
        g.ndata['h'] = node_feats
        out = dgl.max_nodes(g, 'h')
        return out


class MPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MPLayer, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(in_dim, in_dim),
                            nn.BatchNorm1d(in_dim),
                            nn.LeakyReLU(negative_slope=0.2)
                            )

        self.fc2 = nn.Sequential(nn.Linear(in_dim, in_dim),
                            nn.BatchNorm1d(in_dim),
                            nn.LeakyReLU(negative_slope=0.2)
                            )

        self.linear = nn.Linear(in_dim, out_dim)


    def forward(self, g, node_feats):
        g.ndata['h'] = node_feats
        g.send(g.edges(), self.message)
        g.recv(g.nodes(), self.reduce)
        h = g.ndata.pop('h')
        h = self.linear(h)
        return h

    def message(self, edges):
        h = edges.src['h']
        h = self.fc1(h)
        h = self.fc2(h)
        return {'msg': h}

    def reduce(self, nodes):
        return {'h': nodes.mailbox['msg'].mean(1)}

class ZoneEncoder(nn.Module):
    def __init__(self, point_num, out_dim):
        super(ZoneEncoder, self).__init__()
        self.point_num = point_num
        
        self.conv1 = nn.Sequential(nn.Conv1d(10, 64, 1),
                        nn.BatchNorm1d(64),
                        nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv1d(128, 128, 1),
                        )

        self.fc1 = nn.Sequential(nn.Linear(128, 128),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fc2 = nn.Sequential(nn.Linear(128, 128),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fc_final = nn.Sequential(nn.Linear(128, out_dim),
                                   nn.BatchNorm1d(out_dim),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        global_x = torch.max(x, 2, keepdim=True)[0]
        x = torch.flatten(global_x, 1, -1)
        
        x = self.fc1(x)
        x = self.fc_final(x)
        return x
