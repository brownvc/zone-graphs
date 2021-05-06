
import sys
sys.path.append('..')

import os
import argparse
import numpy as np
from dataset import *
from objects import *
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
from evaluation import *
from agent import *
from train_preprocess import *
import hyperparameters as hp
import copy
import joblib

def train_batch(gs, ls, agent):
    loss = agent.update_by_extrusion(ls, gs)
    return loss
    
def train(data_path, folder):

    if os.path.exists(folder) is True: 
        shutil.rmtree(folder) 
    if not os.path.exists(folder):
        os.makedirs(folder)

    agent = Agent(folder)
    
    train_loss_list = []
    validation_loss_list = []
    min_validation_loss = np.inf
    
    gs = []
    ls = []

    for epoch_index in range(hp.train_epoch_num):
        total_loss = 0
        print('epoch', epoch_index , '--------------------------------------------')
        train_ids = read_file_to_list('train_ids.txt')
        for seq_index, seq_id in enumerate(train_ids):
            print('seq index', seq_index, 'seq id', seq_id)
            try:
                gt_seq = data_mgr.load_processed_sequence(os.path.join(data_path, seq_id, 'gt'))
            except:
                continue

            for step_index, gt_step in enumerate(gt_seq):
                try:
                    pos_g = joblib.load(os.path.join(data_path, seq_id, 'train', str(step_index) + '_' + str(1) + '_g.joblib'))
                    pos_e = joblib.load(os.path.join(data_path, seq_id, 'train', str(step_index) + '_' + str(1) + '_e.joblib'))
                    neg_g = joblib.load(os.path.join(data_path, seq_id, 'train', str(step_index) + '_' + str(0) + '_g.joblib'))
                    neg_e = joblib.load(os.path.join(data_path, seq_id, 'train', str(step_index) + '_' + str(0) + '_e.joblib'))
                except:
                    break

                if pos_g and pos_e and neg_g and neg_e:
                    pos_g.encode_with_extrusion(pos_e)
                    gs.append(pos_g)
                    ls.append(to_tensor([1]))

                    neg_g.encode_with_extrusion(neg_e)
                    gs.append(neg_g)
                    ls.append(to_tensor([0]))

                    if len(gs) >= hp.batch_size:
                        loss = train_batch(gs, ls, agent)
                        gs = []
                        ls = []
                        train_loss_list.append(loss)
                        write_list_to_file(os.path.join(folder, 'trainloss.txt'), train_loss_list)
                        agent.save_weights()
 
        # validate after each training epoch
        agent.save_weights()
        validation_loss = validate(data_path, folder)
        validation_loss_list.append(validation_loss)
        write_list_to_file(os.path.join(folder, 'validationloss.txt'), validation_loss_list)
        if validation_loss <= min_validation_loss:
            min_validation_loss = validation_loss
            agent.save_best_weights()

def validate(data_path, folder):

    print('validation----------------------------------------')
    
    agent = Agent(folder)
    agent.load_weights()

    data_mgr = DataManager()

    if os.path.isfile(os.path.join('gt_step_to_extrusions.joblib')):
        step_to_extrusions = joblib.load(os.path.join('gt_step_to_extrusions.joblib'))
    else:
        step_to_extrusions = defaultdict(list)

    total_rank_sum = 0

    validate_ids = read_file_to_list('validate_ids.txt')
    for validation_index, seq_id in enumerate(validate_ids):
        print('validation_index', validation_index, 'seq_id', seq_id)

        try:
            gt_seq = data_mgr.load_processed_sequence(os.path.join(data_path, seq_id, 'gt'))
        except:
            continue
            
        for step_index, gt_step in enumerate(gt_seq):
            try:
                pos_g = joblib.load(os.path.join(data_path, seq_id, 'train', str(step_index) + '_' + str(1) + '_g.joblib'))
                pos_e = joblib.load(os.path.join(data_path, seq_id, 'train', str(step_index) + '_' + str(1) + '_e.joblib'))
                neg_g = joblib.load(os.path.join(data_path, seq_id, 'train', str(step_index) + '_' + str(0) + '_g.joblib'))
                neg_e = joblib.load(os.path.join(data_path, seq_id, 'train', str(step_index) + '_' + str(0) + '_e.joblib'))
            except:
                break

            if pos_g and pos_e and neg_g and neg_e:
                gt_zone_graph = gt_step[0]
                gt_extrusion = gt_step[1]
                extrusions = get_proposals(gt_zone_graph)
                agent_ranked_extrusions = sort_extrusions_by_agent(extrusions, gt_zone_graph, agent)
                for i, extrusion in enumerate(agent_ranked_extrusions):
                    if gt_extrusion.hash() == extrusion.hash():
                        total_rank_sum += i
                        break

    return total_rank_sum

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZoneGraph')
    parser.add_argument('--data_path', default='processed_data', type=str)
    parser.add_argument('--output_path', default='train_output', type=str)
    args = parser.parse_args()

    train(args.data_path, args.output_path)
