# Stair Predictor

Train a instance segmentation model to predict stairs in images.

## Installation

```bash
pip install /home/cjh/workspaces/habitat-explore/third_party/torch-1.10.0+cu111-cp39-cp39-linux_x86_64.whl /home/cjh/workspaces/habitat-explore/third_party/torchvision-0.11.0+cu111-cp39-cp39-linux_x86_64.whl
python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
```

## Usage

File structure:
```bash
├── batch_inference.py
├── coco_3  # coco dataset
├── coco_gene.py  # generate coco dataset
├── configs  # config files for training
├── eval.py  # evaluate the model
├── gene_empty_labelme_json.py  # generate empty labelme json files
├── gene_images.py
├── gene_labels.py  
├── get_images.py  
├── inference.py  # inference the model
├── labelme_json_3  # labelme json files, with or w/o annotations
├── labelme_json_viz_3  # labelme json files visualized
├── labelme_json_viz_processed_3  
├── output_3_add5  # output of the model
├── README.md
├── stair_datasets_3
├── test_images
├── test_images_output
├── train.py
└── vis.sh
```

