import sys
sys.path.append('..')

from objects import *
import random
import torch

def sort_extrusions_by_random(extrusions):
    random.shuffle(extrusions)
    return extrusions

def sort_extrusions_by_heur(extrusions, zone_graph):
    for extrusion in extrusions:
        extrusion.score = get_extrusion_heur_score(extrusion, zone_graph)
        
    sorted_extrusions = sorted(extrusions, key=lambda v: (v.score[0], v.score[1]), reverse=True)
    return sorted_extrusions

def sort_extrusions_by_agent(extrusions, zone_graph, agent):
    with torch.no_grad():
        start_time = time.time()
        if len(extrusions) == 1:
            return extrusions

        g_encs = []
        for extrusion in extrusions:
            zone_graph.encode_with_extrusion(extrusion)
            g_enc = agent.encode_zone_graph(zone_graph)
            g_encs.append(g_enc)
        
        scores = agent.make_decision(g_encs)
        
        for i, extrusion in enumerate(extrusions):
            extrusion.score = scores[i][1].item()

        sorted_extrusions = sorted(extrusions, key=lambda v: v.score, reverse=True)
        return sorted_extrusions
