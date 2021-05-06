import sys
sys.path.append('..')

import numpy as np
import os

import argparse
import numpy as np
from dataset import *
from objects import *
from proposal import *
import multiprocessing
import shutil
import matplotlib.pyplot as plt
import copy
import joblib

def hit_target_in_path(zone_graph, depth, max_depth):
    if zone_graph.is_done():
        return True

    if depth == max_depth:
        return False

    next_extrusions = get_proposals(zone_graph)
    random.shuffle(next_extrusions)
    next_zone_graph = zone_graph.update_to_next_zone_graph(next_extrusions[0])
    ret = hit_target_in_path(next_zone_graph, depth+1, max_depth)
    
    return ret

def generate_neg_steps(pos_steps):
    neg_steps = []
    for i in range(len(pos_steps)):
        time_limit = 100
        start_time = time.time()
        target_depth = len(pos_steps) - i
        sample_number = 10 * target_depth
        base_zone_graph = pos_steps[i][0]
        extrusions = get_proposals(base_zone_graph)
        random.shuffle(extrusions)
        
        min_hit_ratio = 0.999
        best_neg_extrusion = None
        
        print('candidate neg extrusions:', len(extrusions))
        
        for extrusion in extrusions:
            #display_extrusion(extrusion, base_zone_graph, show=True)
            next_zone_graph = base_zone_graph.update_to_next_zone_graph(extrusion)
            hit_count = 0
            for sample_index in range(0, sample_number):
                ret = hit_target_in_path(next_zone_graph, 0, target_depth)
                if ret:
                    hit_count += 1
            hit_ratio = hit_count/sample_number
            print('hit ratio', hit_ratio)
            if hit_ratio <= min_hit_ratio and hit_ratio <= 0.2 and extrusion.hash() != pos_steps[i][1].hash():
                min_hit_ratio = hit_ratio
                best_neg_extrusion = extrusion
                if abs(hit_ratio) < 0.0000001:
                    break
            cur_time = time.time()
            elapsed_time = cur_time - start_time
            print('elapsed_time', elapsed_time)
            if elapsed_time > time_limit:
                break
        if best_neg_extrusion:
            print('neg extrusion found ')
        neg_steps.append((copy.deepcopy(base_zone_graph), copy.deepcopy(best_neg_extrusion)))

    return neg_steps

def process_single_data(seq_id, raw_data_path, processed_data_path):
    data_mgr = DataManager()

    print('precessing episode:', seq_id)

    sequence_length = len(list(Path(os.path.join(raw_data_path, seq_id)).glob('*')))
    print('sequence_length', sequence_length)

    gt_seq = []
    #try:
    gt_seq, error_type = data_mgr.load_raw_sequence(os.path.join(raw_data_path, seq_id), 0, sequence_length)
    if len(gt_seq) == 0:
        return
    #except:
        #return
    
    print('start simulation--------------------------------------------------------------------')
    ret = data_mgr.simulate_sequence(gt_seq)
    print('simulation done------------------------------------------------------------------------------------', ret)
    if not ret:
        return

    seq_gt_folder = os.path.join(processed_data_path, seq_id, 'gt')
    if not os.path.exists(seq_gt_folder):
        os.makedirs(seq_gt_folder)

    for i in range(len(gt_seq)):
        gt_step = gt_seq[i]
        joblib.dump(gt_step[0], os.path.join(seq_gt_folder, str(i) + '_g.joblib'))
        joblib.dump(gt_step[1], os.path.join(seq_gt_folder, str(i) + '_e.joblib'))

    seq_train_folder = os.path.join(processed_data_path, seq_id, 'train')
    if not os.path.exists(seq_train_folder):
        os.makedirs(seq_train_folder)

    step_index = 0
    k = 100
    start_index = 0
    while start_index < sequence_length:
        end_index = min(start_index + k, sequence_length)
        print('start_index', start_index, 'end_index', end_index)

        if sequence_length <= k:
            pos_seq = gt_seq
        else:
            pos_seq = []
            try:
                pos_seq, error_type = data_mgr.load_raw_sequence(os.path.join(raw_data_path, seq_id), start_index, end_index)
                if len(pos_seq) == 0:
                    break
            except:
                break

        start_index += k
        neg_steps = generate_neg_steps(pos_seq)
        
        for i in range(len(neg_steps)):
            pos_step = pos_seq[i]
            #display_zone_graph(pos_step[0], file=os.path.join(seq_train_folder, 'step' + str(step_index) + '_' + str('pos') + '_canvas.png'), show=False)
            #display_extrusion(pos_step[1], pos_step[0], file=os.path.join(seq_train_folder, 'step' + str(step_index) + '_' + str('pos') + '_extrusion.png'), show=False)
            joblib.dump(pos_step[0], os.path.join(seq_train_folder, str(step_index) + '_' + str(1) + '_g.joblib'))
            joblib.dump(pos_step[1], os.path.join(seq_train_folder, str(step_index) + '_' + str(1) + '_e.joblib'))

            neg_step = neg_steps[i]
            #display_zone_graph(neg_step[0], file=os.path.join(seq_train_folder, 'step' + str(step_index) + '_' + str('neg') + 'canvas.png'), show=False)
            #display_extrusion(neg_step[1], neg_step[0], file=os.path.join(seq_train_folder, 'step' + str(step_index) + '_' + str('neg') + 'extrusion.png'), show=False)
            joblib.dump(neg_step[0], os.path.join(seq_train_folder, str(step_index) + '_' + str(0) + '_g.joblib'))
            joblib.dump(neg_step[1], os.path.join(seq_train_folder, str(step_index) + '_' + str(0) + '_e.joblib'))
            
            step_index += 1

    print('single data processing complete !')

def process(raw_data_path, processed_data_path):   

    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    seq_ids = os.listdir(raw_data_path)
    for seq_id in seq_ids:
        sequence_length = len(list(Path(os.path.join(raw_data_path, seq_id)).glob('*')))
        print('sequence_length', sequence_length)

        worker_process = multiprocessing.Process(target=process_single_data, name="process_single_data", args=(seq_id, raw_data_path, processed_data_path))
        worker_process.start()
        worker_process.join(200 + sequence_length * 100)
        
        if worker_process.is_alive():
            print ("process_single_data is running... let's kill it...")
            worker_process.terminate()
            worker_process.join()

    print('all data processing complete !')

def split_data_for_training(dataset_path):
    train_ids = []
    validate_ids = []
    test_ids = []
    all_ids = os.listdir(dataset_path)
    random.shuffle(all_ids)
    length = len(all_ids)
    train_ids = all_ids[0: int(0.85 * length)]
    validate_ids = all_ids[int(0.85 * length) + 1: int(0.9 * length)]
    test_ids = all_ids[int(0.9 * length) + 1: int(1.0 * length)]

    write_list_to_file('train_ids.txt', train_ids)
    write_list_to_file('test_ids.txt', test_ids)
    write_list_to_file('validate_ids.txt', validate_ids)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train_preprocess')
    parser.add_argument('--data_path', default='../data/fusion_processed', type=str)
    parser.add_argument('--processed_data_path', default='processed_data', type=str)
    args = parser.parse_args()

    split_data_for_training(args.data_path)
    process(args.data_path, args.processed_data_path)


