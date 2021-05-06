import sys
sys.path.append('..')
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from objects import *
from models import *
import hyperparameters as hp

def to_numpy(item):
    if item.is_cuda:
        return item.cpu().detach().numpy() 
    else:
        return item.detach().numpy()

def to_tensor(item):
    if torch.cuda.is_available():
        return torch.tensor(item, device=torch.device('cuda:0'), dtype=torch.float)
    else:
        return torch.tensor(item, dtype=torch.float)

class Agent():
    def __init__(self, folder=None):
        self.criterion_agent = nn.NLLLoss()
        self.folder = folder

        self.zone_encoder = ZoneEncoder(zone_sample_num, hp.gnn_node_feat_dim)
        self.zone_encoder.cuda()

        self.decision_maker = GraphNetSoftMax(hp.gnn_node_feat_dim, 2)
        self.decision_maker.cuda()

        self.optim_extrusion = Adam(list(self.zone_encoder.parameters()) + list(self.decision_maker.parameters()), lr=hp.learning_rate_optim_extrusion)

    def encode_zone_graph(self, zone_graph):
        g = dgl.DGLGraph()
        g.add_nodes(len(zone_graph.zone_graph.nodes))

        node_shape_positions = nx.get_node_attributes(zone_graph.zone_graph, 'shape_positions')
        node_shape_normals = nx.get_node_attributes(zone_graph.zone_graph, 'shape_normals')
        node_cur_state_features = nx.get_node_attributes(zone_graph.zone_graph, 'in_current')
        node_tgt_state_features = nx.get_node_attributes(zone_graph.zone_graph, 'in_target')
        node_extru_state_features = nx.get_node_attributes(zone_graph.zone_graph, 'in_extrusion')
        node_bool_features = nx.get_node_attributes(zone_graph.zone_graph, 'bool')
        
        features = []
        target_labels = []
        current_labels = []
        extrusion_labels = []
        bool_labels = []
        shape_positions = []
        shape_normals = []

        point_num = node_shape_positions[0].shape[0]
        
        for i in range(len(zone_graph.zone_graph.nodes)):
            
            cur_label = to_tensor(node_cur_state_features[i])
            current_labels.append(cur_label)

            target_label = to_tensor(node_tgt_state_features[i])
            target_labels.append(target_label)

            extru_label = to_tensor(node_extru_state_features[i])
            extrusion_labels.append(extru_label)

            bool_label = to_tensor(node_bool_features[i])
            bool_labels.append(bool_label)

            shape_position = to_tensor(node_shape_positions[i])
            shape_positions.append(shape_position)

            shape_normal = to_tensor(node_shape_normals[i])
            shape_normals.append(shape_normal)


        current_labels = torch.stack(current_labels)
        current_labels = torch.unsqueeze(current_labels, dim=1)
        current_labels = torch.repeat_interleave(current_labels, point_num, dim=1)
        current_labels = torch.unsqueeze(current_labels, dim=2)

        target_labels = torch.stack(target_labels)
        target_labels = torch.unsqueeze(target_labels, dim=1)
        target_labels = torch.repeat_interleave(target_labels, point_num, dim=1)
        target_labels = torch.unsqueeze(target_labels, dim=2)

        extrusion_labels = torch.stack(extrusion_labels)
        extrusion_labels = torch.unsqueeze(extrusion_labels, dim=1)
        extrusion_labels = torch.repeat_interleave(extrusion_labels, point_num, dim=1)
        extrusion_labels = torch.unsqueeze(extrusion_labels, dim=2)

        bool_labels = torch.stack(bool_labels)
        bool_labels = torch.unsqueeze(bool_labels, dim=1)
        bool_labels = torch.repeat_interleave(bool_labels, point_num, dim=1)
        bool_labels = torch.unsqueeze(bool_labels, dim=2)

        shape_positions = torch.stack(shape_positions)
        shape_normals = torch.stack(shape_normals)

        features = torch.cat((shape_positions, shape_normals, current_labels, target_labels, extrusion_labels, bool_labels), dim=2)
        features = torch.transpose(features, 2, 1)
        encoded_features = self.zone_encoder(features)
        g.ndata['h'] = encoded_features
        
        src = []
        dst = []
        for e in zone_graph.zone_graph.edges:
            src.append(e[0])
            dst.append(e[1])
        src = tuple(src)
        dst = tuple(dst)

        g.add_edges(src, dst)
        g.add_edges(dst, src)

        return g

    def make_decision(self, g_encs):
        bactched_g = dgl.batch(g_encs)
        #bactched_g = torch.stack(g_encs)
        prob = self.decision_maker(bactched_g)
        prob = torch.exp(prob)
        return prob

    def update_by_extrusion(self, labels, gs):
        print('update agent weights')
        self.optim_extrusion.zero_grad()

        g_encs = []
        for i in range(len(gs)):
            g_enc = self.encode_zone_graph(gs[i])
            g_encs.append(g_enc)

        labels = torch.stack(labels)
        labels = labels.long()
   
        prob = self.make_decision(g_encs)        
        gathered_prob = []
        for i in range(len(prob)):
            gathered_prob.append(prob[i][labels[i]])
        prob = torch.stack(gathered_prob)

        gamma = 0.5
        loss = torch.mean(-torch.pow((1 - prob), gamma) * torch.log(prob))
        loss.backward(retain_graph=True)
        self.optim_extrusion.step()

        return loss.item()

    def save_weights(self):
        torch.save(self.zone_encoder.state_dict(), os.path.join(self.folder,"zone_encoder.pkl"))
        torch.save(self.decision_maker.state_dict(), os.path.join(self.folder,"decision_maker.pkl"))

    def save_best_weights(self):
        torch.save(self.zone_encoder.state_dict(), os.path.join(self.folder,"best_zone_encoder.pkl"))
        torch.save(self.decision_maker.state_dict(), os.path.join(self.folder,"best_decision_maker.pkl"))

    def load_weights(self):
        state_dict = torch.load(os.path.join(self.folder,"zone_encoder.pkl"))
        self.zone_encoder.load_state_dict(state_dict)
        state_dict = torch.load(os.path.join(self.folder,"decision_maker.pkl"))
        self.decision_maker.load_state_dict(state_dict)
        
    def load_best_weights(self):
        state_dict = torch.load(os.path.join(self.folder,"zone_encoder.pkl"))
        self.zone_encoder.load_state_dict(state_dict)
        state_dict = torch.load(os.path.join(self.folder,"decision_maker.pkl"))
        self.decision_maker.load_state_dict(state_dict)



