import sys
from setup import *
import FreeCAD
import Part
from FreeCAD import Base
import numpy as np
import json
import copy
from collections import defaultdict
import networkx as nx

from utils.edge_utils import *
from utils.face_utils import *
from utils.solid_utils import *
from utils.vector_utils import *
from utils.vertex_utils import *
from utils.space_splitter import *
from utils.combination_utils import *
from utils.file_utils import *

import time

zone_sample_num = 500
eps = 10e-6
vol_eps = 10e-7

class Extrusion:
    def __init__(self, cad_shape=None):
        self.cad_shape = cad_shape
        self.zone_indices = []
        self.bool_type = None
        self.score = 0

    def copy(self):
        new = Extrusion()
        new.cad_shape = self.cad_shape.copy()
        new.zone_indices = copy.deepcopy(self.zone_indices)
        new.bool_type = copy.deepcopy(self.bool_type)
        return new

    def hash(self):
        return tuple(sorted(self.zone_indices))
    
def get_extrusion_heur_score(extrusion, zone_graph):
    zone_to_current_label = copy.deepcopy(zone_graph.zone_to_current_label)
    zone_to_target_label = zone_graph.zone_to_target_label

    if extrusion.bool_type == 0:
        for i in extrusion.zone_indices:
            zone_to_current_label[i] = True
    else:
        for i in extrusion.zone_indices:
            zone_to_current_label[i] = False

    zone_count = len(zone_graph.zones)
    dismatch_count = 0
    for z_i in range(0, zone_count):
        if zone_to_current_label[z_i] and not zone_to_target_label[z_i]:
            dismatch_count += 1
        if not zone_to_current_label[z_i] and zone_to_target_label[z_i]:
            dismatch_count += 1
    heur_score1 = (zone_count - dismatch_count) / zone_count

    I = 0
    U = 0
    for z_i in range(0, zone_count):
        if zone_to_current_label[z_i] and zone_to_target_label[z_i]:
            U += zone_graph.zones[z_i].cad_shape.Volume
            I += zone_graph.zones[z_i].cad_shape.Volume
        if zone_to_current_label[z_i] and not zone_to_target_label[z_i]:
            U += zone_graph.zones[z_i].cad_shape.Volume
        if not zone_to_current_label[z_i] and zone_to_target_label[z_i]:
            U += zone_graph.zones[z_i].cad_shape.Volume
    heur_score2 = I/U

    heur_score2 = 0
    return heur_score1, heur_score2

class Zone:
    def __init__(self, cad_shape = None):
        self.cad_shape = cad_shape
        self.sample_positions = None
        self.sample_normals = None
        self.score = 0
        self.inside_points = []
    
    def cal_inside_points(self):
        inside_points = []

        tmp = [su.point_inside_solid(self.cad_shape.Solids[0])]
        for v in tmp:
            if v:
                inside_points.append(np.array([v[0], v[1], v[2]]))
            else:
                inside_points.append(None)

        return inside_points

class ZoneGraph:
    def __init__(self, is_forward=True):
        self.current_shape = None
        self.target_shape = None
        self.bbox = None

        #zones
        self.zones = []
        self.zone_graph = nx.Graph()
        self.zone_to_faces = defaultdict(list)
        self.zone_to_current_label = {}
        self.zone_to_target_label = {}
        
        #faces
        self.faces = []
        self.exterior_faces = set()
        self.face_to_zones = defaultdict(list)
        self.face_to_current_label = {}
        self.face_to_target_label = {}
        self.face_to_extrusion_zones = {}
        
        #planes
        self.planes = []
        self.plane_to_faces = defaultdict(list)
        self.plane_to_face_graph = {}
  
        self.plane_to_pos_zones = defaultdict(set)
        self.plane_to_neg_zones = defaultdict(set)

    def build(self, use_face_loop = True):
        if len(self.zones) > 0:
            return False, None

        sp = SpaceSplitter(self.target_shape, use_face_loop = use_face_loop)
        gen_cylinder = sp.isGenCylinder
        
        
        zone_solids = sp.get_zone_solids()
        self.bbox = sp.bbox_used

        print('number of zones', len(zone_solids))
        if len(zone_solids) <= 1 and not gen_cylinder:
            return False, 'single_zone'
        if len(zone_solids) <= 1 and gen_cylinder:
            return False, 'gen_cylinder_single_zone'

        self.planes = sp.get_proposal_planes()
        face_to_index = {}
        for s_i, solid in enumerate(zone_solids):
            for face in solid.Faces:
                key = hash_face(face)
                if key not in face_to_index:
                    face_to_index[hash_face(face)] = len(self.faces)
                    self.faces.append(face)
        
        for f_i, face in enumerate(self.faces):
            for box_face in sp.bbox_geo.Faces:
                if face_share_extension_plane(face, box_face):
                    self.exterior_faces.add(f_i)
                    break
        
        target_volume = 0
        for z_i, zone_solid in enumerate(zone_solids):
            zone = Zone()
            zone.cad_shape = zone_solid
            self.zones.append(zone)
            for face in zone_solid.Faces:
                f_i = face_to_index[hash_face(face)]
                self.zone_to_faces[z_i].append(f_i)
                self.face_to_zones[f_i].append(z_i)

            zone.sample_positions, zone.sample_normals  = get_samples_on_solid_surface(zone.cad_shape, zone_sample_num)
            zone.inside_points = zone.cal_inside_points()

            if solid_contain_zone(self.current_shape, zone):
                self.zone_to_current_label[z_i] = True
            else:
                self.zone_to_current_label[z_i] = False

            if solid_contain_zone(self.target_shape, zone):
                self.zone_to_target_label[z_i] = True
                target_volume += zone.cad_shape.Volume
            else:
                self.zone_to_target_label[z_i] = False

        if abs(target_volume - self.target_shape.Volume) > 0.1 * self.target_shape.Volume:
            return False, 'target_mismatch'

        for p_i, plane in enumerate(self.planes):
            for f_i, face in enumerate(self.faces):
                if face_share_extension_plane(plane, face):
                    self.plane_to_faces[p_i].append(f_i)

        start = time.time()
        
        self.zone_graph = nx.Graph()
        for z_i, zone in enumerate(self.zones):
            self.zone_graph.add_node(z_i)

        visited_faces = set()
        for f_i, face_i in enumerate(self.faces):
            if len(self.face_to_zones[f_i]) == 2:
                z_i = self.face_to_zones[f_i][0]
                z_j = self.face_to_zones[f_i][1]
                self.zone_graph.add_edge(z_i, z_j)
                visited_faces.add(f_i)

        for f_i in range(0, len(self.faces)):
            for f_j in range(f_i, len(self.faces)):
                if f_i not in visited_faces and f_j not in visited_faces:
                    condition = face_touch_condition(self.faces[f_i], self.faces[f_j])
                    if condition > 1:
                        z_i = self.face_to_zones[f_i][0]
                        z_j = self.face_to_zones[f_j][0]
                        self.zone_graph.add_edge(z_i, z_j)
                        # face i contain face j
                        if condition == 2:
                            visited_faces.add(f_j)
                        # face j contain face i
                        if condition == 3:
                            visited_faces.add(f_i)
                        
        print('time building zone graph connections', time.time() - start)

        # build face graph connectivity on each plane
        start = time.time()
        for plane_i, plane in enumerate(self.planes):
            face_graph = nx.Graph()
            visited_edges = set()
            edges = []
            edge_to_index = {}
            edge_to_faces = defaultdict(list)
            for f_i in self.plane_to_faces[plane_i]:
                for edge in self.faces[f_i].Edges:
                    edge_key = hash_edge(edge)
                    if edge_key not in edge_to_index:
                        edge_index = len(edges)
                        edge_to_index[edge_key] = edge_index
                        edges.append(edge)
                    else:
                        edge_index = edge_to_index[edge_key]
                    edge_to_faces[edge_index].append(f_i)

            for e_i in range(len(edges)):
                if len(edge_to_faces[e_i]) == 2:
                    f_i = edge_to_faces[e_i][0]
                    f_j = edge_to_faces[e_i][1]
                    face_graph.add_edge(f_i, f_j)
                    visited_edges.add(e_i)

            for e_i in range(0, len(edges)):
                for e_j in range(e_i, len(edges)):
                    if e_i not in visited_edges and e_j not in visited_edges:
                        condition = edge_touch_condition(edges[e_i], edges[e_j])
                        if condition > 1:
                            f_i = edge_to_faces[e_i][0]
                            f_j = edge_to_faces[e_j][0]
                            
                            # edge i contain edge j
                            if condition == 2:
                                common = edges[e_i].common(edges[e_j])
                                print('common.Length / edges[e_i].Length', common.Length / edges[e_i].Length)
                                if common.Length / edges[e_i].Length >= 0.5:
                                    face_graph.add_edge(f_i, f_j)
                                visited_edges.add(e_j)
                            # edge j contain edge i
                            if condition == 3:
                                common = edges[e_i].common(edges[e_j])
                                print('common.Length / edges[e_j].Length', common.Length / edges[e_j].Length)
                                if common.Length / edges[e_j].Length >= 0.5:
                                    face_graph.add_edge(f_i, f_j)
                                visited_edges.add(e_i)
                        
            self.plane_to_face_graph[plane_i] = face_graph
        
        for plane_i, plane in enumerate(self.planes):
            plane_normal = plane.normalAt(0.5,0.5)
            plane_center = plane.CenterOfMass
            for z_i, z in enumerate(self.zones):
                zone_dir = z.cad_shape.CenterOfMass - plane_center
                if np.dot(zone_dir, plane_normal) >= 0:
                    self.plane_to_pos_zones[plane_i].add(z_i)
                else:
                    self.plane_to_neg_zones[plane_i].add(z_i)

        self.assign_face_labels()
        print('zone graph building complete')
        return True, None

    def assign_face_labels(self):
        for plane_i, plane in enumerate(self.planes):
            plane_normal = plane.normalAt(0.5, 0.5)
            for sign in [1, -1]:
                plane_direction = plane_normal * sign
                face_indices = self.plane_to_faces[plane_i]
                for face_i in face_indices:
                    direction_key = hash_vector(plane_direction)
                    #direction_key = tuple([plane_direction[0], plane_direction[1], plane_direction[2]])
                    #backward_direction_key = tuple([-plane_direction[0], -plane_direction[1], -plane_direction[2]])
                    self.face_to_current_label[(face_i, direction_key)] = False
                    #self.face_to_current_label[(face_i, backward_direction_key)] = False
                    self.face_to_target_label[(face_i, direction_key)] = False
                    #self.face_to_target_label[(face_i, backward_direction_key)] = False
                    for zone_i in self.face_to_zones[face_i]:
                        zone_dir = self.zones[zone_i].cad_shape.CenterOfMass - self.faces[face_i].CenterOfMass
                        zone_dir = zone_dir / np.linalg.norm(zone_dir)
                        if np.dot(plane_direction, zone_dir) > 0:
                            if self.zone_to_current_label[zone_i]:
                                self.face_to_current_label[(face_i, direction_key)] = True
                            else:
                                self.face_to_current_label[(face_i, direction_key)] = False

                            if self.zone_to_target_label[zone_i]:
                                self.face_to_target_label[(face_i, direction_key)] = True
                            else:
                                self.face_to_target_label[(face_i, direction_key)] = False
                    
                #print('pos_count', pos_count)
                #print('neg_count', neg_count)

    def encode_with_extrusion(self, extrusion):

        node_features = {}
        for zone_i, zone in enumerate(self.zones):
            node_features[zone_i] = {}
            shape_positions = zone.sample_positions
            shape_normals = zone.sample_normals

            node_features[zone_i]['shape_positions'] = shape_positions
            node_features[zone_i]['shape_normals'] = shape_normals
            
            if self.zone_to_current_label[zone_i]:
                node_features[zone_i]['in_current'] = 1
            else:
                node_features[zone_i]['in_current'] = 0
            
            if self.zone_to_target_label[zone_i]:
                node_features[zone_i]['in_target'] = 1
            else:
                node_features[zone_i]['in_target'] = 0

            if zone_i in extrusion.zone_indices:
                node_features[zone_i]['in_extrusion'] = 1
            else:
                node_features[zone_i]['in_extrusion'] = 0
        
            node_features[zone_i]['bool'] = extrusion.bool_type

        nx.set_node_attributes(self.zone_graph, node_features)

    def update_to_next_zone_graph(self, extrusion):
        next_zone_graph = ZoneGraph()
        next_zone_graph.bbox = self.bbox
        next_zone_graph.zones = self.zones
        next_zone_graph.faces = self.faces
        next_zone_graph.exterior_faces = self.exterior_faces
        next_zone_graph.planes = self.planes
        next_zone_graph.zone_to_faces = self.zone_to_faces
        next_zone_graph.face_to_zones = self.face_to_zones
        
        next_zone_graph.plane_to_faces = self.plane_to_faces
        next_zone_graph.plane_to_face_graph = self.plane_to_face_graph        
        next_zone_graph.face_to_extrusion_zones = self.face_to_extrusion_zones

        #next_zone_graph.current_shape = self.current_shape
        next_zone_graph.target_shape = self.target_shape
        next_zone_graph.plane_to_pos_zones = self.plane_to_pos_zones
        next_zone_graph.plane_to_neg_zones = self.plane_to_neg_zones


        next_zone_graph.zone_graph = copy.deepcopy(self.zone_graph)
        next_zone_graph.zone_to_current_label = copy.deepcopy(self.zone_to_current_label)
        next_zone_graph.zone_to_target_label = copy.deepcopy(self.zone_to_target_label)

        if extrusion.bool_type == 0:
            for i in extrusion.zone_indices:
                next_zone_graph.zone_to_current_label[i] = True
        else:
            for i in extrusion.zone_indices:
                next_zone_graph.zone_to_current_label[i] = False
        
        #start = time.time()
        next_zone_graph.assign_face_labels()
        #print('time of face label assignment', time.time() - start)
        #next_zone_graph.encode()

        return next_zone_graph

    def get_current_zone_indices(self):
        indices = []
        for zone_i, zone in enumerate(self.zones):
            if self.zone_to_current_label[zone_i]:
                indices.append(zone_i)
        return indices

    def get_target_zone_indices(self):
        indices = []
        for zone_i, zone in enumerate(self.zones):
            if self.zone_to_target_label[zone_i]:
                indices.append(zone_i)
        return indices

    def is_done(self):
        error_volume = 0
        error_count = 0
        threshold_vol = self.target_shape.Volume * 0.01
        #print('threshold_vol', threshold_vol)
        for z_i, zone in enumerate(self.zones):
            if self.zone_to_current_label[z_i] and not self.zone_to_target_label[z_i]:
                error_volume += zone.cad_shape.Volume
                error_count += 1
            if not self.zone_to_current_label[z_i] and self.zone_to_target_label[z_i]:
                error_volume += zone.cad_shape.Volume
                error_count += 1

            if error_volume > threshold_vol:
                #print('error_volume', error_volume, "/",threshold_vol)
                return False
        return True

    def get_IOU_score(self):
        I = 0
        U = 0
        for z_i, zone in enumerate(self.zones):
            if self.zone_to_current_label[z_i] and self.zone_to_target_label[z_i]:
                U += zone.cad_shape.Volume
                I += zone.cad_shape.Volume
            if self.zone_to_current_label[z_i] and not self.zone_to_target_label[z_i]:
                U += zone.cad_shape.Volume
            if not self.zone_to_current_label[z_i] and self.zone_to_target_label[z_i]:
                U += zone.cad_shape.Volume

        return  I/U