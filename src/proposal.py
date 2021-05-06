
import sys
sys.path.append('..')

import numpy as np
from setup import *
from collections import defaultdict
import os
import itertools
import copy
import time
import networkx as nx
import random
from objects import *
import itertools

import networkx as nx


eps = 10e-6
def get_proposals(zone_graph, level=1):
    """
    given zone graph, find valid possible extrusions
    each extrusion is represented as a list of zones (index)
    """
    #print('start proposal')
    start = time.time()
    extrusions = []
    PG = ProposalGenerator(zone_graph)
    combs = PG.get_extrusion_combs(level)
    #print('time of finding proposals', time.time() - start)
   
    for comb in combs:

        all_out_current = True
        all_in_current = True
        for zone_i in comb:
            if PG.zone_graph.zone_to_current_label[zone_i]:
                all_out_current = False
            else:
                all_in_current = False

        #print('all_in_current', all_in_current)
        #print('all_out_current', all_out_current)
        
        bool_types = []
        if all_in_current:
            bool_types = [1]
        else:
            if all_out_current:
                bool_types = [0]
            else:
                bool_types = [0, 1]
        
        for bool_type in bool_types: 
            extrusion = Extrusion()
            extrusion.zone_indices = comb
            extrusion.bool_type = bool_type
            extrusions.append(extrusion)
    
    #print('number of extrusions', len(extrusions))
    return extrusions


def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def get_all_graph_groups(G):
    graphs = list(connected_component_subgraphs(G))
    return [list(g.nodes) for g in graphs]

def get_fixed_size_group_combs(groups, size):
    if size > len(groups):
        return []
    groups = itertools.combinations(groups, size)
    ret_groups = []
    for gs in groups:
        temp_gs = []
        for g in gs:
            temp_gs += g
        ret_groups.append(temp_gs)
    return ret_groups


class ProposalGenerator():
    def __init__(self, zone_graph):
        self.zone_graph = zone_graph
        self.signed_plane_to_face_groups = defaultdict(list)
        self.extrusion_vols = {}
    
    def get_extrusion_combs(self, level=1):
        """
        get all possible extrusion volumes
        """
        eps = 10e-6

        extrusion_combs = []
        visited_extrusions = set()
        
        plane_pairs = get_fixed_size_combinations(range(len(self.zone_graph.planes)), 2)
        validate_pairs = []
        for pair in plane_pairs:
            plane1 = self.zone_graph.planes[pair[0]]
            plane2 = self.zone_graph.planes[pair[1]]
            if face_parallel(plane1, plane2):
                validate_pairs.append((pair[0], pair[1]))
                validate_pairs.append((pair[1], pair[0]))
        
        #print('number of validate pairs', len(validate_pairs))
        start_time = time.time()
        for pair in validate_pairs:            
            start_plane_i = pair[0]
            end_plane_i = pair[1]

            start_plane = self.zone_graph.planes[pair[0]]
            start_normal = start_plane.normalAt(0.5,0.5)

            end_plane = self.zone_graph.planes[pair[1]]
            end_normal = end_plane.normalAt(0.5,0.5)

            p1 = start_plane.CenterOfMass
            p2 = end_plane.CenterOfMass
            d = distance_of_planes(p1, p2, start_normal)
            if abs(d) < eps:
                continue

            extrusion_vector = start_normal * d

            if d < 0:
                signed_start_normal = -1 * start_normal
            else:
                signed_start_normal = start_normal

            loop_to_start = end_plane.CenterOfMass - start_plane.CenterOfMass
            loop_to_end = start_plane.CenterOfMass - end_plane.CenterOfMass

            start_set = None
            end_set = None
            if np.dot(loop_to_start, start_normal) >= 0 and np.dot(loop_to_end, end_normal) >= 0 :
                start_set = self.zone_graph.plane_to_pos_zones[pair[0]]
                end_set = self.zone_graph.plane_to_pos_zones[pair[1]]
            if np.dot(loop_to_start, start_normal) >= 0 and np.dot(loop_to_end, end_normal) < 0 :
                start_set = self.zone_graph.plane_to_pos_zones[pair[0]]
                end_set = self.zone_graph.plane_to_neg_zones[pair[1]]
            if np.dot(loop_to_start, start_normal) < 0 and np.dot(loop_to_end, end_normal) >= 0 :
                start_set = self.zone_graph.plane_to_neg_zones[pair[0]]
                end_set = self.zone_graph.plane_to_pos_zones[pair[1]]
            if np.dot(loop_to_start, start_normal) < 0 and np.dot(loop_to_end, end_normal) < 0 :
                start_set = self.zone_graph.plane_to_neg_zones[pair[0]]
                end_set = self.zone_graph.plane_to_neg_zones[pair[1]]

            middle_zones = list(start_set.intersection(end_set))
            
            signed_plane_key = (pair[0], hash_vector(signed_start_normal))
            if signed_plane_key not in self.signed_plane_to_face_groups:
                
                face_indices = self.zone_graph.plane_to_faces[pair[0]]
                face_graph = self.zone_graph.plane_to_face_graph[pair[0]]

                forward_direction_key = hash_vector(signed_start_normal)
                backward_direction_key = hash_vector(-signed_start_normal)
                
                valid_start_face_indices = []

                #print('face_indices', face_indices)
                
                for face_i in face_indices:
                    current_label_forward = self.zone_graph.face_to_current_label[(face_i, forward_direction_key)]
                    current_label_backward = self.zone_graph.face_to_current_label[(face_i, backward_direction_key)]
                    if current_label_forward and not current_label_backward:
                        valid_start_face_indices.append(face_i)
                        continue
                    if not current_label_forward and current_label_backward:
                        valid_start_face_indices.append(face_i)
                        continue 

                    for exterior_face in self.zone_graph.exterior_faces:
                        exterior_normal = self.zone_graph.faces[exterior_face].normalAt(0.5, 0.5)
                        if 1 - abs(np.dot(self.zone_graph.faces[face_i].normalAt(0.5, 0.5), exterior_normal)) < 10e-6:
                            valid_start_face_indices.append(face_i)
                            break

                #print('len(valid_start_face_indices)', len(valid_start_face_indices))
                    
                if len(valid_start_face_indices) > 0:
                    target_only_faces = []
                    current_only_faces = []
                    idle_faces = []
                    for face_i in valid_start_face_indices:
                        current_label = self.zone_graph.face_to_current_label[(face_i, forward_direction_key)]
                        target_label = self.zone_graph.face_to_target_label[(face_i, forward_direction_key)]
                        
                        if not target_label and current_label:
                            current_only_faces.append(face_i)
                        
                        if target_label and not current_label:
                            target_only_faces.append(face_i)

                        if not target_label and not current_label:
                            idle_faces.append(face_i)

                    #print('target_only_faces', len(target_only_faces))
                    #print('current_only_faces', len(current_only_faces))
                    #print('idle_faces', len(idle_faces))
                    
                    final_current_groups = []
                    if len(current_only_faces) > 0:
                        if len(current_only_faces) == 1:
                            final_current_groups = [current_only_faces]
                        else:
                            current_sub_g = face_graph.subgraph(current_only_faces)
                            current_groups = get_all_graph_groups(current_sub_g)
                            
                            for level_i in range(level):
                                final_current_groups += get_fixed_size_group_combs(current_groups, level_i+1)
                                final_current_groups += get_fixed_size_group_combs(current_groups, len(current_groups)-level_i)
                    
                    final_target_groups = []
                    if len(target_only_faces) > 0:
                        if len(target_only_faces) == 1:
                            final_target_groups = [target_only_faces]
                        else:
                            target_sub_g = face_graph.subgraph(target_only_faces)
                            target_groups = get_all_graph_groups(target_sub_g)
                            
                            for level_i in range(level):
                                final_target_groups += get_fixed_size_group_combs(target_groups, level_i+1)
                                final_target_groups += get_fixed_size_group_combs(target_groups, len(target_groups)-level_i)

                    final_target_idle_groups = []
                    if len(idle_faces + target_only_faces) > 0:
                        if len(idle_faces + target_only_faces) == 1:
                            final_target_idle_groups = [idle_faces + target_only_faces]
                        else:
                            target_idle_sub_g = face_graph.subgraph(idle_faces + target_only_faces)
                            target_idle_groups = get_all_graph_groups(target_idle_sub_g)
                            for level_i in range(level):
                                final_target_idle_groups += get_fixed_size_group_combs(target_idle_groups, level_i+1)
                                final_target_idle_groups += get_fixed_size_group_combs(target_idle_groups, len(target_idle_groups)-level_i)

                    face_groups = final_current_groups + final_target_groups + final_target_idle_groups

                    visited_face_groups = set()
                    for fg in face_groups:
                        if len(fg) > 0:
                            visited_face_groups.add(tuple(sorted(fg)))
                    
                    face_groups = list(visited_face_groups)
                else:
                    face_groups = []
                
                self.signed_plane_to_face_groups[signed_plane_key] = face_groups
            else:
                face_groups = self.signed_plane_to_face_groups[signed_plane_key]

            #print('time spent on face_groups computing', time.time()-start_time)

            #print('number of face groups', len(face_groups))
            start_time = time.time()
            for face_group in face_groups:
                zone_indices = []
                for f_i in face_group:
                    face_key = (f_i, start_plane_i, end_plane_i) 
                    if face_key in self.zone_graph.face_to_extrusion_zones:
                        face_zone_indices = self.zone_graph.face_to_extrusion_zones[face_key]
                    else:
                        solid = self.zone_graph.faces[f_i].copy().extrude(extrusion_vector)
                        face_zone_indices = self.get_inside_zones(self.zone_graph, start_plane_i, end_plane_i, [f_i], middle_zones)
                        # face_zone_indices = self.get_inside_zones_freecad(self.zone_graph, solid)


                        running_vol = 0
                        for z_i in face_zone_indices:
                            zone = self.zone_graph.zones[z_i]
                            running_vol += abs(zone.cad_shape.Volume)
                        
                        if abs(running_vol - solid.Volume) > 0.0001 * solid.Volume:
                            face_zone_indices = []

                        # ensures the result is generalized cylinder
                        # face_zone_indices = self.judge_gen_cyn_zones(face_zone_indices)

                        self.zone_graph.face_to_extrusion_zones[face_key] = face_zone_indices
                    zone_indices += face_zone_indices

                if len(zone_indices) == 0:
                    continue
                
                shape_key = tuple(sorted(zone_indices))
                if shape_key not in visited_extrusions:
                    visited_extrusions.add(shape_key)
                    extrusion_combs.append(zone_indices)
            #print('time spent on iterating face groups', time.time()-start_time)

        return extrusion_combs

    def get_inside_zones(self, zone_graph, start_plane_i, end_plane_i, face_group, zone_candidates):
        start_plane = self.zone_graph.planes[start_plane_i]
        start_normal = start_plane.normalAt(0.5,0.5)

        zones_in_extrusion = set()
        open_faces = list(face_group)
        closed_faces = []
        #print('open_faces', open_faces)
        while len(open_faces) > 0:
            f_to_explore = open_faces[0]
            open_faces.remove(f_to_explore)
            zones = zone_graph.face_to_zones[f_to_explore]
            #print('zones', zones)
            new_zones = set()
            for z_i in zones:
                if z_i in zone_candidates:
                    new_zones.add(z_i)
            
            #print('new zones', new_zones)
            
            zones_in_extrusion = zones_in_extrusion.union(new_zones)
            #print('zones_in_extrusion', zones_in_extrusion)
            #exit()

            new_faces = []
            for z_i in new_zones:
                faces = zone_graph.zone_to_faces[z_i]
                for f_i in faces:
                    if f_i not in open_faces and f_i not in closed_faces:
                        new_faces.append(f_i) 

            for new_f_i in new_faces:
                if abs(np.dot(zone_graph.faces[new_f_i].normalAt(0.5, 0.5), start_normal)) > 10e-3:
                    open_faces.append(new_f_i) 

            closed_faces.append(f_to_explore)

        return list(zones_in_extrusion)