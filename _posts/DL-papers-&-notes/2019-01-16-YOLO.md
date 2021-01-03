#YOLO 1
![](https://upload-images.jianshu.io/upload_images/9165719-76383249e3e45ffd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
契机是因为在写Mask R-CNN的时候，群员突然蹦出一句：

![](https://upload-images.jianshu.io/upload_images/9165719-6bb585324ed91176.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

简单的说下yolo的特点：
1.速度极其快，完全能够实时检测
2.预测精度略低于frcnn

输入：一张图片
输出：所有的边框与对应的分类结果

###过程
1.输入：图片
2.将图片分割成SxS的grid，对于每个grid，预测给出：
B个box，每个box包涵5个信息xyhwc,xy是box的中心xy值与对应grid的偏移量，hw是box的高宽与整张图片高宽的比例值，c是confidence，c:=Pr(obj)*iou，在训练时，当box没有物体时，该值为0，当有物体时，该值等于box与ground truth的iou
C个条件分类概率Pr(Class<sub>i</sub>|Object),代表，在当前grid含有物体情况下，对于大小为C的预测集，每个分类的概率，在验证时，Pr(Class<sub>i</sub>|Object) * Pr(Object) * IOU = Pr(Class<sub>i</sub>)*IOU,即box中的confidence*C中class的概率，能够得到，每个class存在于一个box中的confidence
在PASCAL VOC上训练，S = 7 B =2 ,分类集大小C=20，所以用 S x S x （B x 5 + C）的tensor代表预测结果，7x7x30

###网络设计
![](https://upload-images.jianshu.io/upload_images/9165719-bfc047916ab2897f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
###训练方法
在Imagenet上的1000个分类上对卷积层进行预训练，取图中前20层卷积层，后面跟进avg pooling和fc层，达到了single crop top-5 88% accuracy on ImageNet2012
然后取前20层卷积，跟进4层卷积，2层fc，由于任务从分类变为了目标检测，所以输入从224<sup>2</sup>改为了448<sup>2</sup>

最后一层使用线性激活，其余层使用leaky relu，x<0使用0.1x激活

###LOSS设计
![](https://upload-images.jianshu.io/upload_images/9165719-126e9bb1d5bd7180.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
loss中需要包含的项有，xyhwc，class
图中第一项第二项即是xyhw，第三项第四项是class，但是作者希望，有obj的class loss和noobj的权重应该不一样所以添加了参数项λ<sub>noobj</sub>

#YOLO 9000
