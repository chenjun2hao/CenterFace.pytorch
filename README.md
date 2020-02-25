# the real-time face detection Centerface

unofficial version of centerface, which achieves the best balance between speed and accuracy. Centerface is a practical anchor-free face detection and alignment method for edge devices.

The project provides training scripts, training data sets, and pre-training models to facilitate users to reproduce the results. Finally, thank the centerface's author for the training advice.


## performance results on the validation set of WIDER_FACE 
use the same train dataset without additional data

| Method | Easy | Medium | Hard|
|:--------:| :--------:| :---------:| :------:|
| ours(one scale)| 0.9257 | 0.9131   | 0.7717|
| original | 0.922 | 0.911 | 0.782 |
| ours(multi-scale) | - | - | - |


## Requirements
use pytorch, you can use pip or conda to install the requirements
```sybase
# for pip
cd $project
pip install -r requirements.txt

# for conda
conda env create -f enviroment.yaml
```

## Test
1. download the pretrained model from [Baidu](https://pan.baidu.com/s/1--xWSq5zlvZ-83Y30utI_A) password: z2cs

2. download the validation set of [WIDER_FACE](https://pan.baidu.com/s/1--xWSq5zlvZ-83Y30utI_A) password: z2cs

3. test on the validation set
```sybase
cd $project/src
source activate torch110
python test_wider_face.py
```

4. calculate the accuracy
```sybase
cd $project/evaluate
python3 setup.py build_ext --inplace
python evaluation.py --pred {the result folder}
    
>>>
Easy   Val AP: 0.9257383419951156
Medium Val AP: 0.9131308732465665
Hard   Val AP: 0.7717305552550734
```

5. result
![result](./readme/000388_result.png)

## Train
the backbone use mobilev2 as the same with the original paper
The annotation file is in coco format. the annotation file and train data can download for [Baidu](https://pan.baidu.com/s/1--xWSq5zlvZ-83Y30utI_A) password: z2cs
train
```sybase
cd $project/src/tools
source activate torch110
python main.py
```

## Train on your own data
follow the [CenterNet](https://github.com/xingyizhou/CenterNet)

---

## TO DO
- [ ] use more powerful and small backbone
- [ ] use other FPN tricks



## reference
borrow code from [CenterNet](https://github.com/xingyizhou/CenterNet)
> [**CenterNet**](https://github.com/xingyizhou/CenterNet)  
> [CenterMulti](https://github.com/bleakie/CenterMulti)  
> [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace)

