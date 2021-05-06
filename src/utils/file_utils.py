
import sys
sys.path.append('..')

from setup import *

import FreeCAD
import Part
from FreeCAD import Base
import argparse
import matplotlib.pyplot as plt

def write_val_to_file(val, filename):
    with open(filename, 'w') as f:
        f.write(str(val) + '\n')

def read_file_to_float_value(filename):
    val = None
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            val = float(line)
            return val

def read_file_to_string(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            val = str(line)
            return val

def write_list_to_file(filename, items):
    with open(filename, 'w') as f:
        for item in items:
            f.write(str(item) + '\n')

def append_item_to_file(filename, item):
    with open(filename, 'a') as f:
        f.write(str(item) + '\n')

def read_file_to_list(filename):
    items = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items.append(line.split('\n')[0])
    return items





