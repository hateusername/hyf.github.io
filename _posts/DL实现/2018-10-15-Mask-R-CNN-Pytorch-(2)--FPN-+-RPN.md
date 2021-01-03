![](https://upload-images.jianshu.io/upload_images/9165719-e81a3524fed39b08.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
之前的文章写得太烂了，我自己都看不下去，今天开始重写（2018-12-07）
为了使文章更简洁易懂我决定从输入到输出一步一步探究RPN内部进行的计算

#理论过程与结构
本文的FPN+RPN基于resnet101（resnet50，resnet151，或者resnext都可以作为特征提取网络）

#####过程1.输入图片，获取Resnet特征图
取出图像，输入图像至resnet101,使用hook取出{C<sub>2</sub>，C<sub>3</sub>，C<sub>4</sub>，C<sub>5</sub>}层的feature maps，不选用C<sub>1</sub>层在FPN论文中提到是因为存在large memory footprint，而根据实践来看，本身RPN网络的计算不会消耗过多时间，但是feature maps中数据的存取会消耗很多时间（以至于我真的不想做FPN，尤其是第二阶段的mask判定会消耗更多训练时间）

假设输入原始图片大小为(H,W,3),经过resnet101后
C<sub>2</sub>的feature maps是(H/4,W/4,256)
C<sub>3</sub>为（H/8,W/8,512）
C<sub>4</sub>为（H/16,W/16,1024）
C<sub>5</sub>为（H/32,W/32,2048）
feature maps 的集合称为{C<sub>2</sub>，C<sub>3</sub>，C<sub>4</sub>，C<sub>5</sub>}

#####过程2.特征图处理，生成P
对{C<sub>2</sub>，C<sub>3</sub>，C<sub>4</sub>，C<sub>5</sub>}每层分别使用卷积，生成特征图I，即
I<sub>2</sub> = conv2d（input_dim = 256,output_dim = 256,kernel_size = (1,1））
I<sub>3</sub> = conv2d（input_dim = 512,output_dim = 256,kernel_size = (1,1））
I<sub>4</sub> = conv2d（input_dim = 1024,output_dim = 256,kernel_size = (1,1））
I<sub>5</sub> = conv2d（input_dim = 2048,output_dim = 256,kernel_size = (1,1））

{I<sub>2</sub>，I<sub>3</sub>，I<sub>4</sub>，I<sub>5</sub>}转{M<sub>2</sub>，M<sub>3</sub>，M<sub>4</sub>，M<sub>5</sub>}
为了混合不同层次的特征，将高层进行上采样与低层结合
M<sub>5</sub> = I<sub>5</sub>
M<sub>4</sub> = I<sub>4</sub> + X2I<sub>5</sub>
M<sub>3</sub> = I<sub>3</sub> + X2I<sub>4</sub>
M<sub>2</sub> = I<sub>2</sub> + X2I<sub>3</sub>

由于{M<sub>2</sub>，M<sub>3</sub>，M<sub>4</sub>，M<sub>5</sub>}是由I叠加得来的，为了消除抗混叠效应，对M做一次卷积，生成P
P = conv2d（input_dim = 256,output_dim = 256,kernel = (3,3）)输入为M
额外地下采样（论文中使用maxpooling）生成P<sub>6</sub>

全过程如下
![](https://upload-images.jianshu.io/upload_images/9165719-a1cc1d1acdd725e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
至此已经获得了[P2,P3,P4,P5,P6]

#####过程3.RPN_head扫描[P2,P3,P4,P5,P6]
RPN_head是一个较为简单的层，为nn.Conv2d(256,256,3,padding=1)，将P2P3P4P5P6依次通过后，结果将分别进入cls 和 reg两个子网络用于区域建议

#####过程4.cls_layer与reg_layer分别读取刚才通过RPNhead的信息
cls是nn.Conv2d(256, 6, 1),输入是256维与RPNhead输出对齐，输出是6维,kernel为（1,1）简写为1，分别对应了3个形状的anchor对应的2种概率：形状：[1:1,1:2,2:1] 概率:[P<sub>obj</sub>，P<sub>noobj</sub>]
reg是nn.Conv2d(256,12,1)输入同样与RPNhead对齐，输出是12维对应了3种形状的anchor与4个值[δx,δy,δh,δw]
δx δy是当前anchor中心xy坐标应当上下左右移动的修正量，在这里可以用δx * anchor_w来获得移动距离，当δx为0代表不需要移动
δh δw是伸缩的比例，可以用exp（δh）表示h需要放缩的倍率，δw同理

#####过程5.生成建议区域
结合下文理论1，在P2层的anchor中心坐标对应原图中心坐标，面积对应，长宽比也对应，就能得到面积为32<sup>2</sup>的概率与修正方向
同理，P3层也如此，直到能够获得各种大小各种形状的anchor的[P<sub>obj</sub>，P<sub>noobj</sub>]
此时，将anchor的[x,y,h,w]与对应的reg层[δx,δy,δh,δw]计算获得网络修正后的区域
但是此时，我们是不知道修正后区域的confidence的，我们取与修正区域形状面积最相似的anchor，这个anchor的confidence就是修正区域的confidence，然后我们就有了，建议区域和对应的置信度

#####过程6.选择合适的建议区域作为最终输出
在一张输入图片中，我们很有可能会获得很多建议区域，少则200多则10w+，由于在mask rcnn后阶段训练时，FPN只需要512个区域作为输入，测试时也只需要2k个建议区域，所以在此我们要剔除一些效果不怎么好的建议区域，筛选逻辑为
1.越过边界的区域
2.与其他区域重合过多，同时置信度不高的区域
1.由于最终建议区域是以anchor与δanchor计算生成的区域，会有一定数量的区域最终计算结果超越原图的边界，在此可以直接剔除，或者将稍微越过的部分修正而删除大部分越界的ROI
2.NMS算法筛选
![image.png](https://upload-images.jianshu.io/upload_images/9165719-44fe209af2852e28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
NMS简单而言，对于预测种类相同的区域，取最高置信度区域，其他与其重合度较高的区域全部删除，如上图中，所有红色都被识别为人脸，取置信度最高的区域，删去其他区域
但是在RPN中，我们没有区域的分类信息，所以只需要取置信度最高的区域，重合度较高的区域都可以删去
至此,RPN的工作便全部完成，输出RoI用于mask Rcnn后半部分使用

#####理论1.anchor
anchor有5种尺寸{32<sup>2</sup>，64<sup>2</sup>，128<sup>2</sup>，256<sup>2</sup>，512<sup>2</sup>}
anchor不是一个实体概念！！！anchor并不实际存在于网络结构中，只是一个虚拟的概念。
在RPN_head扫描[P2,P3,P4,P5,P6]时，输入原始图片中32<sup>2</sup>大小的区域映射到P2上是8<sup>2</sup>，以此类推，原图中{32<sup>2</sup>，64<sup>2</sup>，128<sup>2</sup>，256<sup>2</sup>，512<sup>2</sup>}对应在[P2,P3,P4,P5,P6]上面积均为8<sup>2</sup>（anchor在变大，但是P因为Pooling的关系在以相同的比例逐渐缩小）
RPN_head的kernel是（3,3），这时我们虚拟地以RPN_head的kernel中心为锚点，虚拟的建立起3个anchor框，他们的形状为[1:1,1:2,2:1]，他们的面积在P上都是8<sup>2</sup>,但是映射到原始输入图像，P2的8<sup>2</sup>在原图面积为32<sup>2</sup>,P3的8<sup>2</sup>在原图面积为64<sup>2</sup>
而将RPN_head扫描后的信息通过cls reg层进行输出就能获得不同的虚拟框中，概率与修正方向


#RPN训练部分：
#####1.训练算法
使用SGD训练
初始化参数方法在论文中
![faster rcnn原文](https://upload-images.jianshu.io/upload_images/9165719-5433697c6b7b9ea2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#####2.正负样本标定
在训练时，对于某个anchor，我们有ground truth bounding box作为参考目标
当anchor与某个gt box iou > 0.7时将直接标定为正样本，<0.3直接标定为负样本
为了避免某些特殊情况下没有anchor与gt box iou > 0.7 ，我们将iou最大的anchor标定为正

一张图片的训练只使用256个样本，正负比例为1:1，正样本不够负样本来凑，总样本多于256则随机扔掉

#####3.loss
cls的loss就是，正样本尽量接近（1,0）负样本尽量接近（0,1）
比如正样本loss为(P1 - 1)^2 + P2 ^ 2
具体见论文，没有什么难度

#####4.RPN表现
不应当对RPN的表现抱太大期望，LOSS不应当理想，因为RPN的任务就是提供大量的候选区域，LOSS太低你表现这么好后面的cls reglayer就没有了意义

###注意
reg网络的输出不可以是直接的xyhw值！！！
因为仅根据anchor的内容告诉你准确的xyhw是很困难的，而且没有利用anchor的位置信息，anchor的位置是有效的额外信息，利用好会加快收敛速度
预测xyhw网络收敛慢而且会趋近于取图片正中央的anchor，因为这样在概率上loss确实会更低

###错误的实验结果（reg输出直接为xyhw的值）

注意这里给出的bbox是得分非常高的，降低分数要求会使得bbox增加至几万
![](https://upload-images.jianshu.io/upload_images/9165719-d73aa3a5b34e5908.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![140.png](https://upload-images.jianshu.io/upload_images/9165719-9031448f610f5470.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#####智障一样的结果
![仿佛在框背景](https://upload-images.jianshu.io/upload_images/9165719-1e78d13da898559e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

整个实验结果非常差
边缘目标完全无法识别

###部分正确的结果（未NMS,粗略生成）
简单地训练了一下，效果没有很好，主要还是速度不够，RPN本身速度非常快，但是输出数据的处理需要时间
图中只给出了百个bbox，在传输给mask head 时我们要选出大约2000个，基本是能够覆盖所有需要识别的目标
![](https://upload-images.jianshu.io/upload_images/9165719-0831ebb06a58fc99.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/9165719-ba6611a6c8c32552.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


