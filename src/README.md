# ZoneGraphs

### Required Packages
- [Freecad](https://www.freecadweb.org/) 
- [Pytorch](https://pytorch.org/)
- [dgl](https://www.dgl.ai/)
- [mayavi](https://docs.enthought.com/mayavi/mayavi/) (for visualization only)
- [Trimesh]
- [networkx]


### FreeCAD path setup

put absolute path to your FreeCAD lib in setup.py

### Preprocess Fusion360 Raw data to reconstruction sequence data (each step contains current shape, target shape and extrusion shape)

Create "data" folder in the root, put "fusion" and "extrude" in the data folder -> /data/fusion and /data/extrude

python dataset_fusion.py 

--fusion_path "path to your fusion reconstruction data folder"

--extrusion_path "path to your fusion extrusion data folder"

--output_path "path to your output processed fusion data folder"

### Generating training data

python train_preprocess.py

--data_path "path to your processed fusion data folder"

--output_path "path to your output processed data for training"


### Testing/Infering reconstruction sequences

cd experiments/exp_infer

python infer.py

--option "the option for ranking the proposed extrusions: random/heur/agent"

--data_path "path to your processed fusion data folder"

--max_time "time limit for the search to terminate"

--max_step "maximum sequence length"





