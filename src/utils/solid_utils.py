"""
Utility function for solid in 3D
"""

import sys
sys.path.append('..')

from setup import *

import FreeCAD
import Part
from FreeCAD import Base
import trimesh as tm
import numpy as np
import math
import math
import copy
import random
from collections import defaultdict

from utils.face_utils import *
from utils.edge_utils import *
from utils.vertex_utils import *
from utils.combination_utils import *
from utils.vector_utils import *

import time

default_eps = 10e-5

def hash_solid(solid):
    face_keys = []
    for face in solid.Faces:
        face_key = hash_face(face)
        face_keys.append(face_key)
    sorted_face_keys = sorted(face_keys)
    solid_key = tuple(face_key for face_key in sorted_face_keys)
    #solid_key = tuple([solid_key, solid.Volume])
    solid_key = hash(solid_key)
    return solid_key

def solid_contain_zone(solid, zone, eps = default_eps):
    eps = 10e-6
    if solid is None:
        return False
    
    #return su.solid1_contain_solid2(solid, zone.cad_shape)

    for s in solid.Solids:
        for p in zone.inside_points:
            if p is None:
                return False
            if s.isInside(Base.Vector(p[0], p[1], p[2]), 10e-8, True):
                return True        

    return False


def get_gen_cylinder_side_face_loops(all_faces, force_gen_cylinder = False):
    """
    find face loops in cad so that each face loop is a part of a 
    generalized cylinder side faces

    the selected faces for a loops should be:
    - form a loop in face graph
    - their intersection lines are parallel to one another
    """

    edges = []
    edge_to_index = {}
    edge_to_faces = defaultdict(list)
    for f_i,face in enumerate(all_faces):
        for edge in face.Edges:
            edge_key = hash_edge(edge)
            if edge_key not in edge_to_index:
                edge_index = len(edges)
                edge_to_index[edge_key] = edge_index
                edges.append(edge)
            else:
                edge_index = edge_to_index[edge_key]
            edge_to_faces[edge_index].append(f_i)

    """
    - 1. for each face
        create a visited face list, store the current face
        - 2. find parallel edge pairs of the face
            - 3. for each parallel edge pair, find neighboring faces, and record visited faces
                - go back to 2 for each found face
        
        - store found loop (new face found is visited face)
    """

    gen_face_loops = []
    directions = []

    for face_i, face in enumerate(all_faces):

        parallel_edge_pairs = face_get_parallel_edge_pairs(face)
        #print('parallel_edge_pairs', parallel_edge_pairs)
        for pair in parallel_edge_pairs:
            starting_edge = pair[0]
            direction = edge_direction(starting_edge)
            loop = get_next_face(all_faces, face_i, starting_edge, edge_to_index, edge_to_faces,force_gen_cylinder)
            #print('loop', loop)
            if loop and not list_has_list(gen_face_loops, loop):
                gen_face_loops.append(loop)
                directions.append(direction)

    output = [(loop, vector) for (loop, vector) in zip(gen_face_loops,directions)]
    return output


# helper function
def get_next_face(all_faces, current_face_id, starting_edge, edge_to_index, edge_to_faces, force_gen_cylinder = False, face_loop = []):
    if len(face_loop) == 0:
        face_loop = [current_face_id]

    starting_edge_index = edge_to_index[hash_edge(starting_edge)]
    neighbor_face_id = get_neighboring_face_by_edge(current_face_id, starting_edge_index, edge_to_faces)
    
    if neighbor_face_id is None:
        return None

    neighbor_face = all_faces[neighbor_face_id]

    is_frame = face_is_frame(neighbor_face)

    if is_frame:
        return None

    if neighbor_face_id in face_loop:
        return face_loop
    else:
        face_loop.append(neighbor_face_id)

        next_edges = get_parallel_edges(neighbor_face, starting_edge)

        if len(next_edges) == 0:
            return None
        else:
            if force_gen_cylinder:
                if abs(next_edges[0].Length - starting_edge.Length) > 10e-6 * starting_edge.Length:
                    return None

            return get_next_face(all_faces, neighbor_face_id, next_edges[0], edge_to_index, edge_to_faces, force_gen_cylinder, face_loop)

def solid_contain(solid1, solid2, eps = default_eps, count_equal = True):
    """
    solid1 contains solid2
    if count_equal is True, then count equal as contain case
    """

    if solid2.isNull():
        return False

    common_z1z2 = solid1.common(solid2)

    if common_z1z2.isNull():
        return False

    if abs(common_z1z2.Volume - solid2.Volume) < eps * solid2.Volume:
        return True
    # if solid_equal(common_z1z2, solid2, eps):
    #     if count_equal:
    #         return True
    #     else:
    #         return not solid_equal(solid1, solid2, eps)
    else:
        return False


def solid_contain_point(solid, pt, eps = default_eps, count_face = False):
    if pt is None:
        return False

    v_total = true_Volume(solid)
    for s in solid.Solids:
        if s.Volume / v_total > eps:
            if s.isInside(pt, eps, count_face):
                return True        

    return False

def point_inside_solid(solid, eps = default_eps):
    center = solid.CenterOfMass
    # if solid_contain_point(solid, center, 10e-8, True):
    #     return center

    edge_lengths = [e.Length for e in solid.Edges]
    eps_t = 10e-5# * min(edge_lengths)

    # print(solid.Length * 100, center, Base.Vector(0,0,1))


    for i,face in enumerate(solid.Faces):
        direction = face.tangentAt(0.5,0.5)[0]
        cutting_plane = Part.makeCircle(face.Length * 100, center, direction)
        cutting_plane = Part.Wire(cutting_plane)
        cutting_plane = Part.Face(cutting_plane)
        normal = face.normalAt(0.5,0.5)

        cutting_intersection = face.section(cutting_plane)
        # cutting_intersection.exportStep(f"../../../debug{i}.stp")

        for edge in cutting_intersection.Edges:
            edge_c = edge.CenterOfMass
            # edge_c = edge.valueAt(0.5)
            shifted = copy.copy(edge_c).add(copy.copy(normal).multiply(eps))

            line = Part.makeLine(edge_c,shifted)

            # projected point on the face
            near_info = face.distToShape(line)

            for pair in near_info[1]:
                for edge_c_p in pair:
                    face_value = near_info[2][0][2]

                    if type(face_value) is tuple:
                        normal_temp= face.normalAt(face_value[0],face_value[1])
                    else:
                        normal_temp = normal

                    shifted = copy.copy(edge_c_p).sub(copy.copy(normal_temp).multiply(eps_t))
                    # Part.makeSphere(0.001, shifted).exportStep(f"../../../debug{random.random()}.stp")
                    
                    if solid_contain_point(solid, shifted, 10e-8, False):
                        return shifted
        # print('edges', len(cutting_intersection.Edges))

    # for edge in cutting_intersection.Edges:
    #     print('edge center',edge.CenterOfMass)
    # # cutting_edge = cutting_intersection.Edges[0]
    return None

def get_samples_on_solid_surface(solid, amount = 1e2):
    """
    apply uniform sampling on the surface of the solid
    if exclude_hole, 
    """

    mesh = solid.tessellate(default_eps * solid.Length * 100)
    mesh_tri = tm.Trimesh(vertices=mesh[0], faces=mesh[1])
    pos_info, tris = tm.sample.sample_surface(mesh_tri, amount)
    
    positions = []
    normals = []
    for i, p in enumerate(pos_info):
        positions.append(p)
        t = tris[i]
        nor = get_point_normal(p, t, mesh_tri)
        normals.append(nor)
    
    return np.array(positions), np.array(normals)

def solid_is_generalized_cylinder(solid):
    """
    check if there are a pair of parallel faces, then all other faces are perpendicular to this pair of face 
    """
    for i,face1 in enumerate(solid.Faces):
        for j,face2 in enumerate(solid.Faces):
            if i != j and face_parallel(face1, face2):
                
                if abs(face1.Area - face2.Area) < default_eps * solid.Area:
                    # if target_geo:
                    #     # any side is on target surface
                    #     f1 = face_on_solid(target_geo,face1)
                    #     f2 = face_on_solid(target_geo,face2)

                    #     if not f1 and not f2:
                    #         break

                    all_perpendicular = True
                    for k,face_other in enumerate(solid.Faces):
                        if k != i and k != j:
                            if face_perpendicular(face_other, face1) is False:
                                all_perpendicular = False
                                break
                    if all_perpendicular:
                        return face1.normalAt(0.5,0.5)
   
    return False

def merge_solids(solids):

    if len(solids) == 0:
        return None

    merged_solid = solids[0]
    for i in range(1, len(solids)):
        merged_solid = merged_solid.fuse(solids[i])
        #merged_solid = merged_solid.removeSplitter()
    try:
        merged_solid = merged_solid.removeSplitter()
    except:
        return merged_solid
    #merged_solid = defeature_solid(merged_solid)

    #merged_solid = Part.Solid(merged_solid)
    return merged_solid

def get_loop_solid(splitting_shapes):
    compound, map = splitting_shapes[0].generalFuse(splitting_shapes[1:], 0.0)
    if len(compound.Solids) != 2:
        return None
    
    s1 = compound.Solids[0]
    s2 = compound.Solids[1]

    if s1.Volume > s2.Volume:
        return s2
    else:
        return s1

    for s in compound.Solids:
        print('number of faces', len(s.Faces))
        for f in s.Faces:
            print('number of wires', len(f.Wires))

def true_Volume(cad):

    total_vol = 0

    # print('cad', cad.isValid())

    for solid in cad.Solids:
        # print('solid.Volume', solid.Volume)
        total_vol += abs(solid.Volume)
    return max(total_vol,cad.Volume)

def get_bbox(cad):

    bbox = cad.BoundBox

    bbox_data = str(bbox).split('BoundBox (')[1].split(')')[0].split(',')
    bbox_data = [float(item) for item in bbox_data]

    bbox_geo = Part.makeBox(bbox_data[3]-bbox_data[0], bbox_data[4]-bbox_data[1], \
        bbox_data[5]-bbox_data[2], Base.Vector(bbox_data[0], bbox_data[1], bbox_data[2]))
    return bbox_geo