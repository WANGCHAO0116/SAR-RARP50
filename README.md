# Machine Learning Task for PhD Interview at King's College London


## Environment Preparation
1. create a virtual environment using conda and activate it.
```shell
conda create –name openmmlab python=3.8 -y
conda activate openmmlab
```
2. install PyTorch
```shell
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 –extra-index-url https://download.pytorch.org/whl/cu116
```
3. install mmcv
```shell
pip install -U openmim 
mim install mmengine pip install 
mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

## Dataset
### Raw Dataset:

Train set: https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_train_set/24932529

Test set: https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_test_set/24932499

### Data preprocess

A Python script is provided to extract frames from videos at 1 Hz. 

```shell
python sample_video.py -f <num_of_workers> -r <video_dir>
```

### Preprocessed Dataset:

Download link: (coming soon). 

Please put the preprocessed dataset in the directory ```./data```. 

The file structure of data should be as follows: 

```tree
data/
├── images/
│   ├── video_01_000000000.png
│   ├── video_01_000000060.png
│   ├── ...
│   └── video_50_000014520.png
├── labels/
│   ├── video_01_000000000.png
│   ├── video_01_000000060.png
│   ├── ...
│   └── video_50_000014520.png
└── splits/
    ├── train.txt
    └── test.txt
```

## Checkpoints: 

Checkpoints contain some trained weights files. 

Download link: (coming soon). 

Please put the checkpoints in the directory ```./checkpoints```

## Train

Please refer to the file ```train.ipynb``` for the model training code. 

## Inference

We provide two python scripts to make inference for images and videos, respectively. 

### Make inference for image

```shell
python image_inference.py --input_image <input_image_path> --model <model_checkpoint_path>
```

### Make inference for video
```shell
python video_inference.py --input_video <input_video_path> --model <model_checkpoint_path>
```
