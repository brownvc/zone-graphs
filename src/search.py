
import sys
sys.path.append('..')

import numpy as np
import os
import argparse
import numpy as np

from dataset import *
from objects import *
from evaluation import *
from agent import *
from proposal import *
from queue import Queue, LifoQueue, PriorityQueue

import shutil
import time
import random
import copy

# import psutil

class SearchSolution():
    def __init__(self):
        self.ious = []
        self.times = []

        self.best_seq = None
        self.best_time = None
        self.best_score = 0

def dfs_best_recon(zone_graph, max_step, max_time, expand_width, sort_option, best_sol, start_time, folder, agent=None, cur_step=0, cur_seq=[], visited_graphs=set(), visited_extrusions=set(), data_mgr=DataManager()):

    if zone_graph.is_done():
        return True

    if cur_step == max_step :
        return False

    if (time.time() - start_time) > max_time:
        return True

    next_extrusions = get_proposals(zone_graph)
    if len(next_extrusions) == 0:
        return False

    if sort_option == 'agent':
        next_extrusions = sort_extrusions_by_agent(next_extrusions, zone_graph, agent)[0: min(len(next_extrusions), expand_width)]
    else:
        if sort_option == 'heur':
            next_extrusions = sort_extrusions_by_heur(next_extrusions, zone_graph)[0: min(len(next_extrusions), expand_width)]
        else:
            if sort_option == 'random':
                next_extrusions = sort_extrusions_by_random(next_extrusions)[0: min(len(next_extrusions), expand_width)]
            else:
                print('invalid sort option')
                return True
    
    for next_extrusion in next_extrusions:
        next_zone_graph = zone_graph.update_to_next_zone_graph(next_extrusion)
        graph_key = tuple(sorted(next_zone_graph.get_current_zone_indices()))
        if graph_key in visited_graphs:
            continue
        else:
            extrusion_key = tuple(sorted(next_extrusion.zone_indices)) 

            overshadow = False
            for visited_key in visited_extrusions:
                #if set(visited_key).issubset(set(extrusion_key)) or set(visited_key) == set(extrusion_key):
                if set(visited_key) == set(extrusion_key):
                    overshadow = True
                    break
            
            # print('next_extrusion', next_extrusion.zone_indices)
            if overshadow:
                continue
            else:
        
                visited_graphs.add(graph_key)
              
                visited_extrusions.add(extrusion_key)
                cur_seq.append((copy.deepcopy(next_zone_graph), copy.deepcopy(next_extrusion)))
                cur_score = next_zone_graph.get_IOU_score()
             
                if cur_score > best_sol.best_score:
                    print('update best', len(cur_seq), cur_score, best_sol.best_score)
                    best_sol.best_score = cur_score
                    best_sol.best_seq = copy.deepcopy(cur_seq)
                    best_sol.best_time = time.time() - start_time

                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    write_val_to_file(best_sol.best_time, os.path.join(folder, 'time.txt'))
                    write_val_to_file(best_sol.best_score, os.path.join(folder, 'IOU.txt'))
                    data_mgr.save_sequence(best_sol.best_seq, folder)
                
                if cur_step % 2 == 0:
                    expand_width -= 1
                to_exit = dfs_best_recon(next_zone_graph, max_step, max_time, max(1, expand_width), sort_option, best_sol, start_time, folder, agent, cur_step+1, cur_seq, visited_graphs, visited_extrusions, data_mgr)
                cur_seq.pop(-1)

                visited_extrusions.remove(extrusion_key)
                visited_graphs.remove(graph_key)
                
                if to_exit:
                    return True
                
    return False


    
    

        
            


