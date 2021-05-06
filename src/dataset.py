import sys
#sys.path.append('..')

import os
from pathlib import Path
from objects import *
from proposal import *
import json
import glob
import random
import math
import joblib

class Data():
    def __init__(self):
        self.current_shape = Part.Shape()
        self.target_shape = Part.Shape()
        self.extrusion_shape = Part.Shape()
        self.sketch_shape = Part.Shape()
        self.bool_type = 0
    
class DataManager:
    def __init__(self):
        self.dummy = None

    def save_raw_step(self, data, data_path):
        pass

    def load_raw_step(self, data_path):

        data = Data()

        data.target_shape.read(os.path.join(data_path, "target_shape.stp"))

        try:
            data.current_shape.read(os.path.join(data_path, "current_shape.stp"))
            #data.current_shape.scale(100, scale_center)
        except:
            data.current_shape = None

        try:
            data.extrusion_shape.read(os.path.join(data_path, "extrusion.stp"))
            #data.extrusion_shape.scale(100, scale_center)
        except:
            data.extrusion_shape = None

        #data.target_shape.scale(100, scale_center)

        if os.path.isfile(os.path.join(data_path, 'bool_type.txt')):
            ret = read_file_to_string(os.path.join(data_path, 'bool_type.txt'))
            if ret == 'addition':
                data.bool_type = 0
            else:
                data.bool_type = 1
        else:
            data.bool_type = None

        return data

    def check_extrusion_outside(self, extrusion, zone_graph):
        common = extrusion.cad_shape.common(zone_graph.bbox)
        outside = extrusion.cad_shape.cut(zone_graph.bbox)
        #print('outside vol', outside.Volume)
        if outside.Volume > 0.1 * extrusion.cad_shape.Volume:
            return False
        return True

    def check_extrusion_volume(self, extrusion, zone_graph):
        gt_vol = 0
        for solid in extrusion.cad_shape.Solids:
            gt_vol += solid.Volume

        extrusion_vol = 0
        for i in extrusion.zone_indices:
            if solid_contain_zone(extrusion.cad_shape, zone_graph.zones[i]):
                extrusion_vol += zone_graph.zones[i].cad_shape.Volume

        #print('gt_vol', gt_vol)
        #print('extrusion_vol', extrusion_vol)
        if abs(gt_vol - extrusion_vol) > 0.01 * gt_vol:
            print('extrusion vol does not match')
            #display_object(extrusion, zone_graph.bbox, show=True)
            #display_zone_graph(zone_graph)
            inside_zones = []
            for i in extrusion.zone_indices:
                inside_zones.append(zone_graph.zones[i])

            #display_objects(inside_zones, zone_graph.bbox, show=True)
            #print('extrusion vol does not match')
            #exit()
            return False
        return True

    def load_raw_sequence(self, seq_path, min_length=0, max_length=999999):
        
        print('seq_path', seq_path)

        path = Path(seq_path)
        step_names = list(path.glob('*'))
        sequence_length = len(step_names)
        start_index = 0
        max_step = 0
        end_index = sequence_length

        print('step names', step_names)

        for filename in step_names:
            if 'DS' in str(filename):
                os.system("rm -f " + os.path.join(fusion_data_folder, seq_id, str(filename)))

        for step_name in step_names:
            if 'Unsupported' in str(step_name):
                return [], 'unsupported_operation'
            step = int(str(step_name).split('/')[-1])
            if step >= max_step:
                max_step = step

        if len(step_names) != max_step + 1:
            print('length error')
            return [], 'wrong_load_length'

        if sequence_length < min_length or sequence_length > max_length:
            print('length out of bound')
            return [], 'wrong_load_length'
        
        #--------------------------------------------------------------------------------------------------------------------------------
        
        zone_graph = ZoneGraph()

        start_data_path = os.path.join(str(path), str(start_index))
        start_data = self.load_raw_step(start_data_path)

        zone_graph.current_shape = start_data.current_shape
        zone_graph.target_shape = start_data.target_shape


        ret, error_type = zone_graph.build()
        if not ret:
            print('zone graph build error')
            return [], error_type

        sequence = []
        step_indices = list(np.arange(start_index, end_index, 1))
        #print('step_indices', step_indices)
        for step_index in step_indices:
            data_path = os.path.join(str(path), str(step_index))
            data = self.load_raw_step(data_path)
            
            if data.extrusion_shape:
                extrusion = Extrusion(data.extrusion_shape)
                extrusion.bool_type = data.bool_type
                for i, zone in enumerate(zone_graph.zones):
                    if solid_contain_zone(extrusion.cad_shape, zone):
                        extrusion.zone_indices.append(i)

                ret = self.check_extrusion_outside(extrusion, zone_graph)
                if not ret:
                    return [], 'extrusion_outside'

                ret = self.check_extrusion_volume(extrusion, zone_graph)
                if not ret:
                    return [], 'extrusion_mismatch'
                sequence.append((copy.deepcopy(zone_graph), copy.deepcopy(extrusion)))
                next_zone_graph = zone_graph.update_to_next_zone_graph(extrusion)
                zone_graph = next_zone_graph
            else:
                sequence.append((copy.deepcopy(zone_graph), None))

        return sequence, None

    def load_processed_step(self, path):
        graph = joblib.load(str(path) + '_g.joblib')
        extrusion = joblib.load(str(path) + '_e.joblib')
        return (graph, extrusion)

    def load_processed_sequence(self, seq_path, min_length=0, max_length=99999):
        path = Path(seq_path)
        sequence = []
        sequence_length = len(list(path.glob('*_g*')))
        if sequence_length < min_length or sequence_length > max_length:
            print('length out of bound')
            return []
            
        step_indices = list(np.arange(0, sequence_length, 1))
        for step_index in step_indices:
            step_path = os.path.join(seq_path, str(step_index))
            step = self.load_processed_step(step_path)
            sequence.append(step)

        return sequence

    def render_sequence(self, sequence, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, step in enumerate(sequence):
            step_path = os.path.join(path, 'step_' + str(i)) 
            zone_graph = step[0]
            current_zone_shapes = [zone_graph.zones[i].cad_shape for i in zone_graph.get_current_zone_indices()]
            current_shape = merge_solids(current_zone_shapes)
            display_object(current_shape, bound_obj=zone_graph.bbox, color=(0.8, 0.8, 0.8), file=step_path + '_shape.png')

            extrusion = step[1]
            extrusion_zone_shapes = [zone_graph.zones[i].cad_shape for i in extrusion.zone_indices]
            extrusion_shape = merge_solids(extrusion_zone_shapes)
            if extrusion.bool_type == 0:
                display_object(extrusion_shape, bound_obj=zone_graph.bbox, color=(0.0, 1.0, 0), file=step_path + '_extrusion.png')
            else:
                display_object(extrusion_shape, bound_obj=zone_graph.bbox, color=(1.0, 0.0, 0), file=step_path + '_extrusion.png')
    
    def save_sequence(self, sequence, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, step in enumerate(sequence):
            step_path = os.path.join(path, 'step_' + str(i)) 
            zone_graph = step[0]
            current_zone_shapes = [zone_graph.zones[i].cad_shape for i in zone_graph.get_current_zone_indices()]
            current_shape = merge_solids(current_zone_shapes)
            # display_object(current_shape, bound_obj=zone_graph.bbox, color=(0.8, 0.8, 0.8), file=step_path + '_shape.png')

            extrusion = step[1]
            extrusion_zone_shapes = [zone_graph.zones[i].cad_shape for i in extrusion.zone_indices]
            extrusion_shape = merge_solids(extrusion_zone_shapes)

            if not current_shape is None:
                current_shape.exportStep(step_path + '_shape.stp')
            if not extrusion_shape is None:
                extrusion_shape.exportStep(step_path + '_extrusion.stp')
                write_val_to_file(extrusion.bool_type, step_path + '_bool_type.txt')
                
            # if extrusion.bool_type == 0:
            #     display_object(extrusion_shape, bound_obj=zone_graph.bbox, color=(0.0, 1.0, 0), file=step_path + '_extrusion.png')
            # else:
            #     display_object(extrusion_shape, bound_obj=zone_graph.bbox, color=(1.0, 0.0, 0), file=step_path + '_extrusion.png')

    def simulate_sequence(self, seq):
        zone_graph = copy.deepcopy(seq[0][0])
        for i, step in enumerate(seq):
            gt_extrusion = step[1]
            next_extrusion = None
            proposal_extrusions = get_proposals(zone_graph)
            for extrusion in proposal_extrusions:
                if gt_extrusion.bool_type == extrusion.bool_type and gt_extrusion.hash() == extrusion.hash():
                    print('extrusion found')
                    next_extrusion = extrusion
                    break
            
            if next_extrusion is None:
                print('extrusion not found')
                return False

            #display_extrusion(next_extrusion, zone_graph, show=True)
            next_zone_graph = zone_graph.update_to_next_zone_graph(next_extrusion)
            zone_graph = next_zone_graph
            
        #display_object(zone_graph.target_shape, show=True)
        #display_zone_graph(zone_graph, show=True)
        is_done = zone_graph.is_done()
        
        return is_done

