"""
Utility function for edges in 3D
"""

import sys
sys.path.append('..')

from setup import *
import FreeCAD
import Part
from FreeCAD import Base
import random
import copy
import math

from utils.vertex_utils import *

default_eps = 10e-5

def hash_edge(edge, reverse = False):
    parametric_points = edge_sample_points(edge, amount = 50, random_sample = False)
    point_keys = [hash_point(p) for p in parametric_points]
    sorted_point_keys = sorted(point_keys)
    edge_key = tuple(sorted_point_keys)
    edge_key = hash(edge_key)
    return edge_key

def create_segment_edge(pt1, pt2):
    return Part.makeLine(pt1, pt2)

def edge_sample_points(edge, amount = 1, random_sample = True):
    edge_length = edge.Length
    if random_sample:
        params = [random.random() * edge_length for i in range(amount)]
    else:
        #params = [i/(amount-1) * edge_length for i in range(amount-1)] + [ edge_length ]

        lengths = [i / (amount-1) * edge_length for i in range(amount)]
        params = [edge.getParameterByLength(l) for l in lengths]
    #print('params', params)
    return [edge.valueAt(param) for param in params]

def edge_touch_condition(edge1, edge2, eps = default_eps):
    common_e1e2 = edge1.common(edge2)
    if common_e1e2.Length < eps:
        # not touching
        return 0
    else:

        common_length = common_e1e2.Length
        length1 = edge1.Length
        length2 = edge2.Length

        # equal
        if abs(common_length - length1) < eps \
            and abs(common_length - length2) < eps:
            return 1
        
        # face1 contain face2
        if abs(common_length - length1) >= eps \
            and abs(common_length - length2) < eps:
            return 2

        # face2 contain face1
        if abs(common_length - length1) < eps \
            and abs(common_length - length2) >= eps:
            return 3

        # overlap but no contain relationship
        else:
            return 4

def edge_is_straight(edge, eps = default_eps):
    p1 = edge.Vertexes[0].Point
    p2 = edge.Vertexes[-1].Point

    return abs(copy.deepcopy(p1).sub(p2).Length - edge.Length) < eps * edge.Length

def edge_direction(edge, common = False):
    p1 = edge.Vertexes[0].Point
    p2 = edge.Vertexes[-1].Point

    v = (copy.deepcopy(p2).sub(p1))
    
    if v.Length > 0:
        v.normalize()

    if common:
        return (v.x, v.y, v.z)

    else:
        return v

def edge_parallel(edge1, edge2, eps = default_eps):

    if len(edge1.Vertexes) < 2 or len(edge2.Vertexes) < 2:
        return False
    
    if not edge_is_straight(edge1) or not edge_is_straight(edge2):
        return False
    
    v1 = edge_direction(edge1)
    v2 = edge_direction(edge2)

    ang = v1.getAngle(v2)

    if ang < eps or abs(ang - math.pi) < eps:
        return True
    else:
        return False
