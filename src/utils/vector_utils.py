import numpy as np
import copy
import math

def parallel(v1, v2, default_eps = 10e-6):
    # not parallel
    ang = v1.getAngle(v2)

    if abs(ang) > default_eps and abs(ang - math.pi) > default_eps:
        return False
    return True

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    

    rotation_matrix = np.column_stack((rotation_matrix, np.array([0, 0, 0]) )) 
    rotation_matrix = np.vstack ((rotation_matrix, np.array([0, 0, 0, 1]) ))

    return rotation_matrix.flatten()

def vector_dir_hashed(vector):
    x = hash(round(vector[0], 3))
    y = hash(round(vector[1], 3))
    z = hash(round(vector[2], 3))
    x_ = hash(round(-vector[0], 3))
    y_ = hash(round(-vector[1], 3))
    z_ = hash(round(-vector[2], 3))
    if 100*x + 10*y + z > 100*x_ + 10*y_ + z_ :
        vector_key = hash(tuple([x, y, z, x_, y_, z_]))
        return vector_key
    else:
        vector_key = hash(tuple([x_, y_, z_, x, y, z]))
        return vector_key

def hash_vector(vector):
    x = hash(round(vector[0], 3))
    y = hash(round(vector[1], 3))
    z = hash(round(vector[2], 3))
    vector_key = hash(tuple([x, y, z]))
    return vector_key

def vector_rev_equal(vector1, vector2):
    equal = abs(vector1[0] - vector2[0]) + abs(vector1[1] - vector2[1]) + abs(vector1[2] - vector2[2]) < 10e-5
    rev = abs(vector1[0] + vector2[0]) + abs(vector1[1] + vector2[1]) + abs(vector1[2] + vector2[2]) < 10e-5
    return equal or rev