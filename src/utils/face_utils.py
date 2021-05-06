"""
Utility function for faces in 3D
"""

import sys
sys.path.append('..')

from setup import *
import FreeCAD
import Part
from FreeCAD import Base

import math
import random as random
import numpy as np

import copy
from collections import defaultdict

from utils.vector_utils import *
from utils.vertex_utils import *
from utils.edge_utils import *
from utils.combination_utils import *

default_eps = 10e-5

def face_is_planar(face, eps = default_eps):

    c1,c2 = face.curvatureAt(0.5,0.5)

    if abs(c1) + abs(c2) < default_eps: 
        return abs(face.Volume) < eps * face.Area
    else:
        return False

def hash_face(face):
    edge_keys = []
    for edge in face.Edges:
        edge_key = hash_edge(edge)
        edge_keys.append(edge_key)
    sorted_edge_keys = sorted(edge_keys)
    face_key = tuple(edge_key for edge_key in sorted_edge_keys)
    face_key = hash(face_key)
    return face_key

def face_equal(face1, face2, eps = default_eps):
    """
    face1 and face2 geometrically equal
    """

    # shortcut
    if abs(face1.Area - face2.Area) > eps:
        return False

    common_f1f2 = face1.common(face2)

    if abs(common_f1f2.Area - face1.Area) < eps \
        and abs(common_f1f2.Area - face2.Area) < eps:
        return True
    else:
        return False

def pts_on_face(face, point_list, eps = default_eps):
    """
    all points in list are on face
    """
    for pt in point_list:
        if not face.isInside(pt, eps, True):
            return False
    return True

def face_touch_condition(face1, face2, eps = default_eps):
    common_f1f2 = face1.common(face2)
    if common_f1f2.Area < eps:
        # not touching
        return 0
    else:

        common_area = common_f1f2.Area
        area1 = face1.Area
        area2 = face2.Area

        # equal
        if abs(common_area - area1) < eps \
            and abs(common_area - area2) < eps:
            return 1
        
        # face1 contain face2
        if abs(common_area - area1) >= eps \
            and abs(common_area - area2) < eps:
            return 2

        # face2 contain face1
        if abs(common_area - area1) < eps \
            and abs(common_area - area2) >= eps:
            return 3

        # overlap but no contain relationship
        else:
            return 4

def face_parallel(face1, face2, eps = default_eps):
    """
    face1 and face2 are planar faces
    """
    if not face_is_planar(face1) or not face_is_planar(face2):
        return False

    dir1 = face1.normalAt(0.5,0.5)
    dir2 = face2.normalAt(0.5,0.5)
    ang = dir1.getAngle(dir2)

    if abs(ang)<eps or abs(ang - math.pi)<eps:
        return True

    else:
        return False

def face_perpendicular(face1, face2, eps = default_eps):
    """
    face1 and face2 are planar faces
    """

    dir1 = face1.normalAt(0.5,0.5)
    dir2 = face2.normalAt(0.5,0.5)
    ang = dir1.getAngle(dir2)

    if abs(ang - math.pi * 0.5) > eps:
        return False
    else:
        return True

def face_share_extension_plane(face1, face2, eps = 10e-5):

    if not face_parallel(face1, face2, eps):
        return False

    p1 = face1.CenterOfMass
    p2 = face2.CenterOfMass

    n1 = face1.normalAt(0.5,0.5)

    if abs(distance_of_planes(p1,p2,n1)) < eps:
        return True

    return False

def distance_of_planes(point1, point2, normal):
    """
    find the distance between two parallel planes, signed by normal direction.
    normal is the normal vector of plane1.
    point1 and point2 are points on plane1 and plane2 
    """
    A = normal.x
    B = normal.y
    C = normal.z
    x1 = point1.x
    y1 = point1.y
    z1 = point1.z
    x2 = point2.x
    y2 = point2.y
    z2 = point2.z

    D1 = - A * x1 - B * y1 - C * z1
    D2 = - A * x2 - B * y2 - C * z2

    d = abs(D2 - D1)/math.sqrt(A * A + B * B + C * C)

    # apply sign based on normal direction
    dir = point2.sub(point1)
    ang = abs(normal.getAngle(dir))
    if ang > math.pi/2:
        d = -d

    return d

def selective_extend_face(face_id, face_loops, all_faces, scale = 300):
    """
    if face not in face_loops, extend it using extend_face function.
    otherwise, only extend it along loop direction
    """

    face = all_faces[face_id].copy()

    if not face_is_planar(face):
        return extend_face(face, scale)
    

    inloop = False
    scaled_dirs = []

    for lst, direction in face_loops:
        if face_id in lst:
            hashed_dir = vector_dir_hashed(direction)
            if hashed_dir in scaled_dirs:
                continue
            else:
                inloop = True 
                scaled_dirs.append(hashed_dir)
                face = extend_face_along_direction(face, direction, scale)

    if inloop:
        return face
    else:
        return extend_face(face, scale)
    

def extend_face(face_, scale_ = 300):
    """
    extend face by scale
    will scale up using only the outer wire of the face.
    """
    face = face_.copy()
    scale = face.Length * scale_

    # is not flat face
    if not face_is_planar(face):

        sph_data = is_spherical_face(face)
        # return full sphere
        if sph_data:
            c,r = sph_data
            # print(c,r)
            return Part.makeSphere(abs(r), c)
        
        cyl_data = is_cylindrical_face(face)

        # return extended cylinder
        if cyl_data:
            d,f = cyl_data
            extrude_v = copy.copy(d).multiply(scale)
            translate_v = copy.copy(d).multiply(scale/2 * -1)

            f.translate(translate_v)

            c1 = f.extrude(extrude_v)

            combined = c1
            faces = combined.Faces
            # areas = [f.Area for f in faces]
            # index = areas.index(max(areas))

            return faces[0]
        # return original face
        face.scale(1 + 10e-4, face.CenterOfMass)

        return face
    else:
        circle = Part.makeCircle(scale, face.CenterOfMass, face.normalAt(0.5, 0.5))

        wire = Part.Wire(circle)
        f = Part.Face(wire)
        return f

def is_cylindrical_face(face, sample_density = 4):
    """
    the given face is cylindrical
    if is cylindrical face, return its extend direction and one extrude end face
    else return False
    """

    c1_ = None
    c2_ = None

    if face_is_planar(face):
        return False

    for u in range(sample_density):
        for v in range(sample_density):
            u = u/(sample_density-1)
            v = v/(sample_density-1)
            c1,c2 = face.curvatureAt(u,v)

            if c1_ is None:
                c1_ = c1
            if c2_ is None:
                c2_ = c2
                        
            if abs(c1_ - c1) > default_eps\
                or abs(c2_ - c2) > default_eps\
                    or abs(c1 * c2) > default_eps\
                        or (abs(c1) < default_eps and abs(c2) < default_eps):
                return False
    
    tan = face.tangentAt(0.5,0.5)

    if abs(c1) < default_eps:
        extrude_dir = tan[0]
    else:
        extrude_dir = tan[1]
    

    for edge in face.Edges:

        cutting_plane = Part.makeCircle(face.Length * 100, edge.CenterOfMass, extrude_dir)
        cutting_plane = Part.Wire(cutting_plane)
        cutting_plane = Part.Face(cutting_plane)

        cutting_intersection = face.section(cutting_plane)

        cutting_edge = cutting_intersection.Edges[0]

        if cutting_edge.Length > face.Length * default_eps:
            if not edge_is_straight(cutting_edge):
                center = cutting_edge.centerOfCurvatureAt(0.5 * cutting_edge.Length)
                r = 1/ (max(abs(c1_), abs(c2_)))
                # r = r * (1 + 10e-4) 

                return extrude_dir,Part.makeCircle(r, center, extrude_dir)
            
            else:
                return extrude_dir,cutting_edge

def is_spherical_face(face, sample_density = 4):
    """
    the given face is spherical or part of spherical face
    if is spherical face, return its center location and radius
    else return False
    """

    c1_ = None
    c2_ = None

    for u in range(sample_density):
        for v in range(sample_density):
            u = u/(sample_density-1)
            v = v/(sample_density-1)
            c1,c2 = face.curvatureAt(u,v)

            if c1_ is None:
                c1_ = c1
            if c2_ is None:
                c2_ = c2

            if abs(c1_ - c1) > default_eps\
                or abs(c2_ - c2) > default_eps\
                    or abs(c1 - c2) > default_eps\
                        or abs(c1) < default_eps:
                return False
    
    r = 1/c1

    normal = face.normalAt(0.5,0.5)
    direction = copy.copy(normal).multiply(r)
    pt = face.valueAt(0.5,0.5)
    center = copy.copy(pt).add(direction)

    return center,r


def extend_face_along_direction(face, extrude_dir, scale_=300):
  
    scale = scale_ * face.Length
    # only apply directional extend for planar faces
    if not face_is_planar(face):
        return extend_face(face, scale_) 
    
    cutting_plane = Part.makeCircle(face.Length * 10, face.CenterOfMass, extrude_dir)
    cutting_plane = Part.Wire(cutting_plane)
    cutting_plane = Part.Face(cutting_plane)

    projected_vs = [v.Point.projectToPlane(face.CenterOfMass, extrude_dir) for v in face.Vertexes]

    v1 = projected_vs[0]


    dists = [copy.copy(v).sub(v1).Length for v in projected_vs]
    v2 = None

    for i,dist in enumerate(dists):
        if dist > 10e-6:
            v2 = projected_vs[i]
            break
    direction = copy.copy(v2).sub(v1)
    direction.normalize()

    dists_signed = []
    for i,d in enumerate(dists):
        if d < 10e-6:
            dists_signed.append(0)
        else:
            dists_signed.append(d * copy.copy(projected_vs[i]).sub(v1).normalize().dot(direction))
    
    min_pos = dists_signed.index(min(dists_signed)) 
    max_pos = dists_signed.index(max(dists_signed)) 

    end1 = projected_vs[min_pos]
    end2 = projected_vs[max_pos]

    projected_face = create_segment_edge(end1, end2)

    extrude_v = copy.copy(extrude_dir).multiply(scale)
    translate_v = copy.copy(extrude_dir).multiply(scale/2 * -1)

    projected_face.translate(translate_v)
    extended = projected_face.extrude(extrude_v)

    out = extended.Faces[0]
    out.scale(1 + 10e-2, out.CenterOfMass)

    return out

def face_get_parallel_edge_pairs(face):
    
    edge_pairs = get_fixed_size_combinations(range(len(face.Edges)), 2)

    validate_pairs = []

    for pair in edge_pairs:
        edge1 = face.Edges[pair[0]]
        edge2 = face.Edges[pair[1]]

        if edge_parallel(edge1, edge2):
            validate_pairs.append((edge1, edge2))
            # validate_pairs.append((pair[1], pair[0]))
    
    return validate_pairs


def get_neighboring_face_by_edge(face_id, edge_index, edge_to_faces):

    faces_on_edge = edge_to_faces[edge_index]

    if face_id in faces_on_edge:
        for f_id in faces_on_edge:
            if f_id != face_id:
                return f_id
    return None

def get_parallel_edges(face, edge):
    found = []
    key = hash_edge(edge)
    for e in face.Edges:
        temp_key = hash_edge(e)
        if key != temp_key:
            if edge_parallel(edge, e):
                found.append(e)
    
    return found

def face_is_frame(face, threshold_ratio = 0.8, threshold_count = 4):
    """
    the input face:
    - is planar
    - has and only has 1 inner loop
    - inner loop length/outer loop length is larger than threshold
    """

    if not face_is_planar(face):
        return False

    wires = face.Wires

    if len(wires[0].Edges) > threshold_count:
        return True
    
    # curves = []
    for edge in face.Edges:
        if not edge_is_straight(edge):
            # curves.append(edge)
            return True
    
    # if len(curves) == 1 or len(curves) == 3 or len(curves) == 4:
    #     return True
    
    # if len(curves) == 2:
    #     if abs(curves[0].Length - curves[1].Length) > default_eps * curves[1].Length:
    #         return True
        
    #     if 

    if len(wires) > 1:
        outer = wires[0]
        inner = wires[1]

        if inner.Length / outer.Length > threshold_ratio:
            return True
        else:
            return False
