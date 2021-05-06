

import sys
sys.path.append('..')

from setup import *

import FreeCAD
import Part
from FreeCAD import Base
import math
import numpy as np

def hash_vertex(vertex, eps = 3):
    x = hash(round(vertex.Point[0], eps))
    y = hash(round(vertex.Point[1], eps))
    z = hash(round(vertex.Point[2], eps))
    vertex_key = hash(tuple([x, y, z]))
    return vertex_key

def hash_point(point, eps = 3):
    x = hash(round(point[0], eps))
    y = hash(round(point[1], eps))
    z = hash(round(point[2], eps))
    point_key = hash(tuple([x, y, z]))
    return point_key

def get_point_normal(point, face_index, mesh):

    f = mesh.faces[face_index]

    normal_0 = mesh.vertex_normals[f[0]]
    normal_1 = mesh.vertex_normals[f[1]]
    normal_2 = mesh.vertex_normals[f[2]]

    p0 = mesh.vertices[f[0]]
    p1 = mesh.vertices[f[1]]
    p2 = mesh.vertices[f[2]]

    area_2 = get_area(point, p0, p1)
    area_0 = get_area(point, p1, p2)
    area_1 = get_area(point, p2, p0)

    weight_0 = area_0 / (area_0 + area_1 + area_2)
    weight_1 = area_1 / (area_0 + area_1 + area_2)
    weight_2 = area_2 / (area_0 + area_1 + area_2)

    point_normal = normal_0 * weight_0 + normal_1 * weight_1 + normal_2 * weight_2

    return normalize(point_normal)

def get_area(p0, p1, p2):

    a = cal_vec_length(p0 - p1)
    b = cal_vec_length(p1 - p2)
    c = cal_vec_length(p2 - p0)
    p = (a + b + c) / 2

    tmp = p * (p - a) * (p - b) * (p - c)

    tmp = max(0, tmp)

    area = math.sqrt(tmp)
    return area

def cal_vec_length(vec):

    sum = 0
    for v in vec:
        sum += v * v

    return math.sqrt(sum)

def normalize(vec):
    return vec / np.linalg.norm(vec)