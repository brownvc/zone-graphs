

import sys
sys.path.append('..')

import os
from pathlib import Path
import json
import glob
import random
import math
import joblib

from objects import *
from dataset import *

class FusionDataManager:

    def __iter__(self):
        return self

    def __init__(self, data_path, shuffle = True, start_index = 0, augment = False, reposition = False):
        self.num = start_index
        self.augment = augment
        self.reposition = reposition

        print('data_path', data_path)

        file_list = glob.glob(data_path + "/*.step")
        print(f'[DATA LOADER] Found {len(file_list)} models')
        file_list.sort()

        found_sequences = []
        current_sequence = []

        for i, f_dir in enumerate(file_list):
            if i < 10e10:
                f_name = f_dir.split('/')[-1]
                segments = f_name.split("_")

                # found target model
                if len(segments) == 3:
                    # current_sequence.append(f_dir)
                    if len(current_sequence) >= 1 :
                        found_sequences.append(current_sequence)

                    current_sequence = []
                
                # found sequence model of a target model
                else:
                    current_sequence.append(f_dir)
        if len(current_sequence) >= 1:
            found_sequences.append(current_sequence)
        


        print('found_sequences', found_sequences)
        self.pairs = []
        # find modeling pairs
        for seq in (found_sequences):
            target = seq[-1]

            # (current, next, possible target, final target, index, temp_target_count)

            if self.augment:
            # augment data
                for j,temp_target in enumerate(seq[1:]):
                    self.pairs.append((None, seq[0], temp_target, target, f'0_{str(j)}', len(seq[1:]) ))

                for i in range(len(seq)-1):
                    for j,temp_target in enumerate(seq[i+1:]):
                        self.pairs.append((seq[i], seq[i+1], temp_target, target, f'{str(i+1)}_{str(j)}', len(seq[i+1:])))
            else:
            # un-augmented data
                self.pairs.append((None, seq[0], target, target, "0", 1 ))
                for i in range(len(seq)-1):
                    self.pairs.append((seq[i], seq[i+1], target, target, f'{str(i+1)}', 1))

        
        if shuffle:
            random.shuffle(self.pairs)
        
        print(f'[DATA LOADER] Found {len(self.pairs)} extrusion pairs')
        self.pair_count = len(self.pairs)

        print('self.pairs', self.pair_count)


    def get_data_by_pair(self, pair):
        current_dir = pair[0]
        next_dir = pair[1]
        temp_target_dir = pair[2]
        final_target_dir= pair[3]
        index = pair[4]

        count = pair[5]

        segs = next_dir.split('/')[-1].split('_')
        file_name = '_'.join([segs[0],segs[1],segs[2]])

        data = Data()
        data.file_name = file_name
        data.index = index

        data.count = count
        data.target_shape.read(temp_target_dir)

        # calculate normalize scale factor
        final_cad_shape = Part.Shape()
        final_cad_shape.read(final_target_dir)
        bbox = final_cad_shape.BoundBox
        bbox_data = str(bbox).split('BoundBox (')[1].split(')')[0].split(',')
        bbox_data = [float(item) for item in bbox_data]
        w = bbox_data[3]-bbox_data[0]
        d = bbox_data[4]-bbox_data[1]
        h = bbox_data[5]-bbox_data[2]
        x = (bbox_data[3]+bbox_data[0])/2
        y = (bbox_data[4]+bbox_data[1])/2
        z = (bbox_data[5]+bbox_data[2])/2

        diagnal_d = math.sqrt(w*w + d*d + h*h)
        scale_factor = 1 / diagnal_d
        move_vector = Base.Vector(-x, -y, -z)

        if current_dir is None:
            data.current_shape = None

            next_cad_shape = Part.Shape()
            next_cad_shape.read(next_dir)
            data.extrusion_shape = None
            data.bool_type = 0

        else:
            # NOTE: currently extrusion is the subtraction of target and current 
            data.current_shape.read(current_dir)

            next_cad_shape = Part.Shape()
            next_cad_shape.read(next_dir)

            data.extrusion_shape = None
            if su.true_Volume(next_cad_shape) > su.true_Volume(data.current_shape):
                data.bool_type = 0
            else:
                data.bool_type = 1

        data.target_shape.scale(scale_factor, Base.Vector(0, 0, 0))
        if data.current_shape:
            data.current_shape.scale(scale_factor, Base.Vector(0, 0, 0))

            # normalize model location
            if self.reposition:
                data.current_shape.translate(move_vector)

        return data
    
    def __next__(self):
        if self.num < len(self.pairs):
            num = self.num
            picked = self.pairs[num]
            self.num += 1

            res = self.get_data_by_pair(picked)
            if res:
                return num, res
            else:
                # NOTE: when one of the model pairs is not valid, return None

                return num, None
        else:
            raise StopIteration

def preprocess_fusion_data(fusion_path, extrusion_path, processed_fusion_path, reposition):
    
    # process current shapes
    dm = FusionDataManager(fusion_path, shuffle = False, start_index=0, augment=False, reposition=reposition)
    if not os.path.exists(processed_fusion_path):
        os.makedirs(processed_fusion_path)  
    
    for i, data in dm:
        if data:
            print(f"------------Preparing steps {i}/{dm.pair_count}------------")

            file_path = os.path.join(processed_fusion_path, data.file_name)
            try:  
                os.mkdir(file_path)  
            except OSError as error:  
                pass

            step_path = os.path.join(file_path, str(data.index))

            try:  
                os.mkdir(step_path)  
            except OSError as error:  
                pass
            
            if data.current_shape and not data.current_shape.isNull():
                data.current_shape.exportStep(f"{str(step_path)}/current_shape.stp")
            
            data.target_shape.exportStep(f"{str(step_path)}/target_shape.stp")

            if data.bool_type == 0:
                f = open(f"{str(step_path)}/bool_type.txt","w+")
                f.write('addition')
                f.close()
            else:
                f = open(f"{str(step_path)}/bool_type.txt","w+")
                f.write('subtraction')
                f.close()

    file_list = glob.glob(os.path.join(processed_fusion_path, "*"))

    # process extrusions
    for i,file in enumerate(file_list):
        model_id = file.split(processed_fusion_path)[1]

        model_dir = f"{fusion_path}{model_id}.step"

        final_cad_shape = Part.Shape()
        final_cad_shape.read(model_dir)
        bbox = final_cad_shape.BoundBox
        bbox_data = str(bbox).split('BoundBox (')[1].split(')')[0].split(',')
        bbox_data = [float(item) for item in bbox_data]
        w = bbox_data[3]-bbox_data[0]
        d = bbox_data[4]-bbox_data[1]
        h = bbox_data[5]-bbox_data[2]
        x = (bbox_data[3]+bbox_data[0])/2
        y = (bbox_data[4]+bbox_data[1])/2
        z = (bbox_data[5]+bbox_data[2])/2

        diagnal_d = math.sqrt(w*w + d*d + h*h)
        scale_factor = 1 / diagnal_d
        move_vector = Base.Vector(-x, -y, -z)

        exts = glob.glob(extrusion_path + f"{model_id}*.step")
        exts.sort()

        print('Processing Extrusion: ', i, model_dir)

        for j,ext in enumerate(exts):
            ext_shape = Part.Shape()
            ext_shape.read(ext)

            # normalize model location
            if (reposition):
                ext_shape.translate(move_vector)

            # normalize model scale
            ext_shape.scale(scale_factor, Base.Vector(0, 0, 0))

            out_dir = f"{file}/{j}/extrusion.stp"
            try:
                ext_shape.exportStep(out_dir)
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_path', default='../data/fusion/reconstruction', type=str)
    parser.add_argument('--extrusion_path', default='../data/extrude', type=str)
    parser.add_argument('--processed_fusion_path', default='../data/fusion_processed', type=str)
    parser.add_argument('--reposition', default=False, type=bool)

    args = parser.parse_args()

    preprocess_fusion_data(args.fusion_path, args.extrusion_path, args.processed_fusion_path, args.reposition)
