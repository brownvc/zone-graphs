# ZoneGraphs

### Required Packages
- [Freecad](https://www.freecadweb.org/) 
- [Pytorch](https://pytorch.org/)
- [dgl](https://www.dgl.ai/)
- [mayavi](https://docs.enthought.com/mayavi/mayavi/) (for visualization only)
- [Trimesh]
- [networkx]


### FreeCAD path setup

Please download and compile FreeCAD and put absolute path to your FreeCAD lib in setup.py

### Dataset

We use the [Fusion360GalleryDataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset) for training. Sepecifically, we use the reconstruction subset of the Fusion dataset. The download links for the data we use are:
- [Main reconstruction subset](https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/reconstruction/r1.0.0/r1.0.0.zip)
- [GT extrusion set (extrude tools )](https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/reconstruction/r1.0.0/r1.0.0_extrude_tools.zip)

### Preprocess Fusion360 Raw data to reconstruction sequence data (each step contains current shape, target shape and extrusion shape)

Downloaded the above fusion and extrude_tool 
```
python dataset_fusion.py 

--fusion_path "path to your fusion reconstruction data folder"

--extrusion_path "path to your fusion GT extrusion tool data folder"

--output_path "path to your output processed fusion data folder"
```
### Generating training data
```
python train_preprocess.py

--data_path "path to your processed fusion data folder"

--output_path "path to your output processed data for training"
```

### Testing/Infering reconstruction sequences
```
cd experiments/exp_infer

python infer.py

--option "the option for ranking the proposed extrusions: random/heur/agent"

--data_path "path to your processed fusion data folder"

--max_time "time limit for the search to terminate"

--max_step "maximum sequence length"
```




