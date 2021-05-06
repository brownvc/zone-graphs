from mayavi.mlab import *
from mayavi import mlab
import random
import numpy as np
import os
import utils.edge_utils as eu
import utils.solid_utils as su
import utils.face_utils as fu

import networkx as nx

cyan = (0.0, 1.0, 1.0)
yellow = (1.0, 1.0, 0)
red = (1, 0, 0)
green = (0, 1, 0) 

def display_cad(cad_shape, color = (1,1,0), opacity=0.5, show_lines = True):

    default_eps = 10e-5
    triangles = cad_shape.tessellate(default_eps * cad_shape.Length * 10)
    vertex_positions = triangles[0]
    face_indices = triangles[1]

    if len(face_indices) == 0:
        return

    x = []
    y = []
    z = []
    for v in vertex_positions:
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])

    tris = []
    for f in face_indices:
        tri = []
        tri.append(f[0])
        tri.append(f[1])
        tri.append(f[2])
        tris.append(tri)

    mlab.triangular_mesh(x, y, z, tris, color = color, opacity=opacity)

    if show_lines:
        for edge in cad_shape.Edges:
            points = eu.edge_sample_points(edge, amount = 2000, random_sample = False)
            x = []
            y = []
            z = []
            for p in points:
                x.append(p[0])
                y.append(p[1])
                z.append(p[2])
            mlab.points3d(x, y, z, color = (0, 0, 0), scale_factor=0.0025)


def display_object(obj, bound_obj=None, file=None, show=False, export_file=None, color=(1.0, 0.1, 0.1), opacity=0.9):

    if obj is None:
        return

    mlab.figure(bgcolor=(1,1,1), size=(800, 800))
    if bound_obj:
        display_cad(bound_obj)

    if obj:
        if hasattr(obj, 'cad_shape'):
            display_cad(obj.cad_shape, color=color, opacity=opacity)
        else:
            display_cad(obj, color=color, opacity=opacity)

    if export_file:
        if hasattr(obj, 'cad_shape'):
            obj.cad_shape.exportStep(export_file)
        else:
            obj.exportStep(export_file)

    if file:
        mlab.savefig(filename=file)
        
        mlab.close()

    if show:
        mlab.show()

def display_objects(objs, bound_obj=None, file=None, show=False, color=(1.0, 0.1, 0.1), opacity=1.0):

    mlab.figure(bgcolor=(1,1,1), size=(800, 800))

    if bound_obj:
        display_cad(bound_obj)

    for obj in objs:
        if obj:
            if hasattr(obj, 'cad_shape'):
                display_cad(obj.cad_shape, color=color, opacity=0.9)
            else:
                for solid in obj.Solids:
                    display_cad(solid, color=color, opacity=0.9)

    if file:
        mlab.savefig(filename=file)
        mlab.close()
    
    if show:
        mlab.show()

def display_step(extrusion_shape, ext_type, original_shape, bound_obj=None, file=None, show=False):

    mlab.figure(bgcolor=(1,1,1), size=(800, 800))

    if bound_obj:
        display_bound_obj(bound_obj)
    
    if extrusion_shape and not extrusion_shape.isNull():
        if ext_type == "add":
            display_cad(extrusion_shape, color=(0.0, 1, 0.0), opacity=0.5)
        elif ext_type == "remove":
            display_cad(extrusion_shape, color=(1, 0.0, 0.0), opacity=0.5)
        else:
            display_cad(extrusion_shape, color=(0.0, 0.0, 1), opacity=0.5)

    if original_shape and not original_shape.isNull():
        # for solid in original_shape.Solids:
        display_cad(original_shape, color=(0.95, 0.95, 0.95), opacity=0.2)

    if file:
        mlab.savefig(filename=file)
        mlab.close()
    
    if show:
        mlab.show()



    


    




    