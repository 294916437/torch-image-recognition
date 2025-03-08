## about the repository

Several advanced training programs about deep learning and image recognition,which are mainly based on torch and focus on Chinese traditional medicine images.

## local running environment

- anaconda python 310
- CUDA Tech(test.py in root dir is used to checkout your environment,which output would be like below or more advanced device)
  ```
    2.0.1+cu117
    11.7
    True
    1
    NVIDIA GeForce RTX 3050 Laptop GPU
  ```

#### packages version

```
Package                 Version
----------------------- ------------
Jinja2                  3.1.5
mpmath                  1.3.0
networkx                3.4.2
numpy                   1.26.4
packaging               24.2
pandas                  2.2.3
pillow                  11.0.0
pip                     25.0
pyparsing               3.2.1
pytz                    2025.1
requests                2.28.1
scipy                   1.15.2
seaborn                 0.13.2
sympy                   1.13.3
tensorboard             2.19.0
tensorboard-data-server 0.7.2
threadpoolctl           3.5.0
torch                   2.0.1+cu117
torchvision             0.15.2+cu117
typing_extensions       4.12.2
tzdata                  2025.1
urllib3                 1.26.13
```

## program structure

matchimage/
├── data/ # 数据集文件夹
│ └── Medicine/ # 中医药图像数据集
├── runs/ # TensorBoard 日志
├── medicine_train.py # 训练脚本
├── medicine_val.py # 验证与预测脚本
├── medicine_best_model.pth # 训练的最佳模型
└── README.md # 项目说明

#### dataset structure

data/
└── Medicine/
├── train/
│ ├── 三七/
│ │ ├── image001.jpg
│ │ └── ...
│ ├── 人参/
│ │ └── ...
│ └── ...
└── test/
├── 三七/
│ └── ...
└── ...

## methods

#### commands

**train model**

```
python medicine_train.py
```

**validate model**

```
python medicine_val.py --image ./val-data/medicine/甘草\_0.jpg --model medicine_final_model.pth
```

**show training records**

```
tensorboard --logdir=runs
```

## Thanks!!!

- pytorch teams
