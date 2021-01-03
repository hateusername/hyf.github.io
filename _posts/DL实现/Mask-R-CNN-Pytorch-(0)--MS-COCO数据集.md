####基本思路：
先训练resnet101（现成的） + RPN + FPN网络，给出ROI
然后训练接下来的class,mask,box分支网络

数据集选定的时候没有太多讲究吧，选用MS COCO
RPN网络需要训练生成ROI，分支网络需要mask，class，box
这些信息都包含在MS COCO中

#####第一步数据集内容选定
选择train images作为训练集，标注集使用trainval annotation 中的instances_train2017文件夹
train_images包含18G共118287张图片
instances_train2017包含所有图片的标注

#####第二步提取数据
标注文件instances_train2017.json大小448M
为字典信息，索引关键字为
![](https://upload-images.jianshu.io/upload_images/9165719-0d0da37a64f73210.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

######没有用的info
```
"info": {
        "description": "COCO 2017 Dataset", # 数据集描述
        "url": "http://cocodataset.org", # 下载地址
        "version": "1.0", # 版本
        "year": 2017, # 年份
        "contributor": "COCO Consortium", # 提供者
        "date_created": "2017/09/01" # 数据创建日期
    },
```
######没有用的license
```
"licenses": [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        ……
        ……
    ],
```
######images
有用的部分为图片名，高，宽，ID编号（图片名和ID在annotation中也会有，高宽可以在输入图片的时候读取信息，所以也不是必要的信息）
```
"images": [
        {
            "license": 4,
            "file_name": "000000397133.jpg", # 图片名
            "coco_url":  "http://images.cocodataset.org/val2017/000000397133.jpg",# 网路地址路径
            "height": 427, # 高
            "width": 640, # 宽
            "date_captured": "2013-11-14 17:02:52", # 数据获取日期
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",# flickr网路地址
            "id": 397133 # 图片的ID编号（每张图片ID是唯一的）
        },
        ……
        ……
    ],
```
######category
category我们可以单独提取出来作为新的文件（原文件较大较杂）这包含了RPN网络训练时需要的class数据
```
"categories": [ # 类别描述
        {
            "supercategory": "person", # 主类别
            "id": 1, # 类对应的id （0 默认为背景）
            "name": "person" # 子类别
        },
        {
            "supercategory": "vehicle", 
            "id": 2,
            "name": "bicycle"
        },
        {
            "supercategory": "vehicle",
            "id": 3,
            "name": "car"
        },
        ……
        ……
    ],
```
######annotation
1.首先看最后一项，id，这是对象的ID，独立于image_id，因为每一张图片可能有很多物体会标记出来，所以annotation中可能会有两项image_id相同，但id一定不同
2.然后，对于每个标记的物体，有iscrowd = 0,1两种状态，0代表标记物体是单个存在的，1代表标记的物体是集群存在的
3.单个与集群会影响到segmentation的标记方式
4.iscrowd = 0时，segmentation使用边界多边形框选mask，依次为x1 y1 x2 y2 x3 y3的值，由于可能有遮挡的问题，可能会存在多个多边形标记一个物体的情况，所以segmentation可能包含多个元素
5.iscrowd = 1时，由于集群，边界多边形可能边太多而导致记法复杂占用高，使用RLE记法（自行百度RLE游程编码）：M = [0,0,0,1,1,1,1,1,1,0,0]，则M的RLE编码为[3,6,2]，例如M=[0 0 1 1 1 0 1]， RLE就是[2 3 1 1]；M=[1 1 1 1 1 1 0]， RLE为[0 6 1]，注意奇数位始终为0的个数。另外，也使用一个基于LEB128的通用方案的可变比特率来完成额外的压缩。
annotation中我们需要的信息是
segmentation：mask
iscrowd：用于判定segmentation记法
image_id:图片ID
bbox:bounding box用于训练RPN生成ROI
category_id:class
id:对象id用于区分对象
```
"annotation": [
        {
            "segmentation": [ # 对象的边界点（边界多边形）
                [
                    224.24,297.18,# 第一个点 x,y坐标
                    228.29,297.18, # 第二个点 x,y坐标
                    234.91,298.29,
                    ……
                    ……
                    225.34,297.55
                ]
            ],
            "area": 1481.3806499999994, # 区域面积
            "iscrowd": 0, # 
            "image_id": 397133, # 对应的图片ID（与images中的ID对应）
            "bbox": [217.62,240.54,38.99,57.75], # 定位边框 [x,y,w,h]
            "category_id": 44, # 类别ID（与categories中的ID对应）
            "id": 82445 # 对象ID，因为每一个图像有不止一个对象，所以要对每一个对象编号（每个对象的ID是唯一的）
        },
        ……
        ……
        ]
```

#####第三步生成数据
你可以用自己的方法生成尽量精简的数据用于训练特定的网络
RPN需要的标注存在json中大概可以控制在35M左右
Mask R-CNN需要的标注可以控制在400M左右
完整的标注集当然也可以，我就当练练手处理一下json文件了
注意！不要！为了美观而在文件中输出过多的换行符，json.dump（...indent=4）时408M的内容能够变成1.4G

在此我们就完成了数据集的基本处理，可以开始训练RPN了
