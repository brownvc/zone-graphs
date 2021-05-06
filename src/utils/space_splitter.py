"""
class used to partition the 3D space by the extended faces of a 
given geometry.

"""


# path to your FreeCAD.so or FreeCAD.dll file
# make sure to run the python compatible with FreeCAD

import sys
sys.path.append('..')

from setup import *


import FreeCAD
import Part
from FreeCAD import Base

import utils.face_utils as fu
import utils.solid_utils as su
import utils.combination_utils as cu
# from utils.vis_utils import *

dafault_eps = 5 * 10e-4

partition_eps = 10e-9

class SpaceSplitter:
    def __init__(self, geometry, extend_cull = None, use_face_loop = True):
        """
        class to split space (geometry bbox) by extended faces of the geometry 
        extend_cull is a list to specify faces to extend/not to extend
        """

        #self.geometry = geometry.removeSplitter()
        self.geometry = geometry
        self.isGenCylinder = su.solid_is_generalized_cylinder(self.geometry)

        self.bbox = self.geometry.BoundBox
        self.bbox_used = None

        # bbox_data = str(self.bbox).split('BoundBox (')[1].split(')')[0].split(',')
        # bbox_data = [float(item) for item in bbox_data]

        # self.bbox_geo = Part.makeBox(bbox_data[3]-bbox_data[0], bbox_data[4]-bbox_data[1], \
        #     bbox_data[5]-bbox_data[2], Base.Vector(bbox_data[0], bbox_data[1], bbox_data[2]))

        self.bbox_geo = su.get_bbox(geometry)

        # self.bbox_geo.exportStep(f"../../../bbox.stp")

        # resolve tangent touching spliting bug

        self.faces = self.geometry.Faces

        # remove irregular small faces
        self.faces = [f for f in self.faces if f.Area > 10e-8 * self.geometry.Area]

        self.zones = None
        
        use_face_loop = True
        if use_face_loop:
            self.face_loops = su.get_gen_cylinder_side_face_loops(self.faces)
        else:
            self.face_loops = []
        print('len(self.face_loops)', len(self.face_loops))
        
        #print('self.face_loops', self.face_loops, len(self.face_loops))

        if not extend_cull or len(extend_cull) == 0:
            self.splitting_faces = [fu.selective_extend_face(i, self.face_loops, self.faces) for i in range(len(self.faces))]
        else:
            # only extend faces that are true in extend_cull
            self.splitting_faces = [fu.selective_extend_face(i, self.face_loops, self.faces) if (i < len(extend_cull) and \
                extend_cull[i]) or i >= len(extend_cull) else f for i,f in enumerate(self.faces)]


        # print('self.splitting_faces', len(self.splitting_faces))
        self.selected_list = []
        
        for i,face in enumerate(self.splitting_faces):
            # planar extends
            # if len(face.Edges) == 1 and fu.face_is_planar(face):
            if len(face.Edges) == 1 or fu.face_is_planar(face):

                if cu.list_has_item(self.bbox_geo.Faces[:], face, fu.face_share_extension_plane):
                    # self.splitting_faces.remove(face)
                    # face.exportStep(f"{'../../../'}/face_{i}.stp")

                    # print('removed', i)
                    pass
                else:
                    self.selected_list.append(face)
            else:
                self.selected_list.append(face)
        
        # filter co-planar faces
        self.selected_final = []
        planar_ext_faces = []
        non_exts = []

        for face in self.selected_list:
            # planar extensions
            if len(face.Edges) == 1 and face.Area > 10e3:
                # print('area',face.Area, len(face.Edges))
                planar_ext_faces.append(face)
            else:
                non_exts.append(face)
        
        
        # only keep one if two faces are co-planar
        for face in planar_ext_faces:
            if not cu.list_has_item(self.selected_final, face, fu.face_share_extension_plane):
                self.selected_final.append(face)

        # if non-extended is coplanar to planar one, then remove it
        for face in non_exts:
            if not fu.face_is_planar(face):
                self.selected_final.append(face)
            else:
                if not cu.list_has_item(planar_ext_faces, face, fu.face_share_extension_plane):
                    self.selected_final.append(face)



        # print('self.splitting_faces', len(self.splitting_faces))
        # self.faces[4].exportStep(f"{'../../../'}/face_4_nonextend.stp")
        # self.splitting_faces[4].exportStep(f"{'../../../'}/face_4_extend.stp")

        self.list_of_shapes_list = []

        scales = [1 - 1e-5]
        self.bboxs = []
        for scale in scales:
            bbox_temp = self.bbox_geo.copy()
            bbox_temp.scale(scale, bbox_temp.CenterOfMass)
            item = [bbox_temp] + self.selected_final
            self.bboxs.append(bbox_temp)
            self.list_of_shapes_list.append(item)

        # for f in self.splitting_faces:
        #     print('f area',f.Area)

        # faces = Part.makeCompound(self.splitting_faces)
        # faces.exportStep(f"../../../res_faces.stp")

        # print('self.splitting_faces', len(self.splitting_faces))

    def get_zone_solids(self, export_path=None):


        for i,list_of_shapes in enumerate(self.list_of_shapes_list):

            if len(self.geometry.Faces) == 6 and abs(self.geometry.Volume - self.bbox_geo.Volume) < 10e-6:
                self.zones = self.geometry.Solids
                self.bbox_used = self.bboxs[i]
                self.list_of_shapes = list_of_shapes
                break
            else:
                try:
                    # pieces receives a compound of shapes; map receives a list of lists of shapes, defining list_of_shapes <--> pieces correspondence
                    pieces, map = list_of_shapes[0].generalFuse(list_of_shapes[1:], partition_eps)

                    if (len(pieces.Solids) > 1) or i == len(self.list_of_shapes_list) - 1:
                        self.zones = pieces.Solids
                        self.bbox_used = self.bboxs[i]
                        self.list_of_shapes = list_of_shapes
                        break
                except:
                    self.zones = self.geometry.Solids
                    self.bbox_used = self.bboxs[i]
                    self.list_of_shapes = list_of_shapes
                    break
        
        if self.isGenCylinder:
            if len(self.zones) == 1:
                self.zones = self.geometry.Solids
                self.bbox_used = self.bbox_geo
                self.list_of_shapes = self.bbox_geo.copy()


        if export_path:
            self.export_zones(export_path)
        return self.zones

    def get_proposal_planes(self):

        planes = []
        for f in self.faces:
            if fu.face_is_planar(f) and not cu.list_has_item(planes + self.bbox_geo.Faces, f, fu.face_share_extension_plane):
                planes.append(f.copy())

        return planes + self.bbox_used.Faces

    def export_zones(self, path):
        if self.zones is None:
            self.get_zone_solids(None)

        solids = Part.makeCompound(self.zones)

        # export bbox of the geometry
        self.bbox_geo.exportStep(f"{path}/res_bbox.stp")

        # export partitioned space using the geometry
        solids.exportStep(f"{path}/res_p.stp")

        faces = Part.makeCompound(self.list_of_shapes)
        faces.exportStep(f"{path}/res_faces.stp")






