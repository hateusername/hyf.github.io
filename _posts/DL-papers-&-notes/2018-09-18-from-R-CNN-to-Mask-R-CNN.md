论文阅读顺序
 [R-CNN](http://fcv2011.ulsan.ac.kr/files/announcement/513/r-cnn-cvpr.pdf)   ，[fast R-CNN](https://arxiv.org/abs/1504.08083)   ，[faster R-CNN](https://arxiv.org/abs/1506.01497)   ，[Mask R-CNN](https://arxiv.org/abs/1703.06870)

文章简化了很多网络实现细节，因为你们不可能只参考我的文章就去复现成果，显然是要去读原论文的
所以在此就简单地谈谈各个网络的功能与基本实现思路，忽略具体的实现方法

![](https://upload-images.jianshu.io/upload_images/9165719-4d9db5cf6784eaa1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
各个论文的核心任务：
R-CNN：不同尺寸的输入图像输入，能够输出图像中物体的分类与位置信息，即输出上图第三张图片
fast R-CNN：比R-CNN更快完成任务
faster R-CNN：比更快还快
mask R-CNN：能够完成以上所有任务，并且输出第四张图片，同时速度相对faster R-CNN而言不明显降低

####R-CNN，region-based CNN
R-CNN作为2012-2014年左右的研究成果，水平已经不能和当前研究比较，但是却是当前所有架构的基础
2012年ILSVRC比赛中 Krizhevsky等人用CNN在物体识别领域展现了CNN惊人的准确度，在这个背景下R-CNN诞生

一个很自然的想法是CNN卷积层能够提取图像的结构信息
我们利用现有的网络提取结构信息，再进行简单的分类，便可以完成分类的任务
但是如何获得具体的位置信息并没有很好的办法，传统的计算机思路便是随便列举足够多的框，效果不好便扔掉

所以R-CNN的思路很简单：
0.选一个现成的识别率较高的CNN，微调使得输出固定为20类（尽量减少需要识别的内容，减轻任务负担）
1.用selective search算法生成足够多的候选框（一般是2千个）
2.将这些候选框全部通过CNN，取合适的卷积层信息通过FC提取特征向量
3.将这些特征向量输入单独训练的许多SVM，SVM的任务是用来判定，是不是狗，是不是猫，是不是鸡，简单的二分问题
4.如果候选框合适，则输出候选框位置以及SVM判定的结果，如果候选框不那么合适，就使用box regression修改候选框直到大小位置合适

R-CNN的问题：
0.selective search生成2000个候选框本身速度并不差，但是每一个候选框都要单独通过CNN前向传播获得结构信息，使得计算成本太高
1.2000个候选框通过CNN的信息，太多，只能先从运算设备中取出来再存储，存取过程会耗费不少时间
2.由于使用了FC层提取特征向量，FC层的输入是固定的（卷积层对输入大小没有要求)，为了使得FC层正常工作，只能将候选框拉伸或压缩成固定的大小输入卷积层，但是拉伸图片本身会对图像内容造成形变，卷积层只能保证平移不变性，不能保证拉伸等等形变下的不变性（就好像说你把脸横向拉开放到分类网络里面我觉得直觉上比原来正常的脸更容易被判定成猪）
3.可能是当时对CNN不那么信任的问题，仍然在使用SVM进行最终分类，而没有像今天那么推崇end to end的训练方式
![](https://upload-images.jianshu.io/upload_images/9165719-a14a93615e0814eb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


####fast R-CNN
fast R-CNN就觉得2000多个候选框单独通过CNN太麻烦了，消耗太大
同时又不想伸缩图片造成信息破坏而引入了从SPP衍生的RoI Pooling方法

RoI pooling是一种对任意尺寸输入，能够得到固定尺寸输出的池化方法
简单而言就是，我要求输出是10x5的尺寸，你给我100x50的图片，那我就把100x50切割成10x5的小块，每一块都有10x10个像素点，取这10x10中最大值作为池化输出即可（可能产生非整数，对输出取整，这对图像分类影响较小，所以不考虑由此带来的误差）

整个fast R-CNN 的思路也很清晰：
0.selective search生成的2000多个候选框不要单独通过，完整的图片直接一次性通过CNN，把候选框的位置直接射到feature map上
1.把feature map上候选区域经过RoI Pooling生成固定大小的特征量
2.特征量通过FC层，后进入两个子网络，一个用于分类，一个用于边框回归
![](https://upload-images.jianshu.io/upload_images/9165719-01337f6231df6214.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
fast R-CNN的加速表现非常好，让人们意识到实时的物体检测是可行的（对于数据量不大的图片，每秒3帧的计算速度）
![](https://upload-images.jianshu.io/upload_images/9165719-340b03133403e368.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####Faster R-CNN
猜测：随着对神经网络理解的加深，很多情况下研究者会更倾向于end to end，即直接给输入，直接给输出，希望中途所有的任务都由神经网络来完成

faster R-CNN就希望，也别搞什么selective search了干脆让神经网络自己去想办法计算候选区域
核心就是RPN + fast R-CNN

RPN，region proposal network，用于生成候选区域的网络
RPN网络就是简单的给出了几个固定尺寸的以滑动窗口为中心的候选区域，训练过程中，由于有大量ground truth数据用于训练，使得RPN本身并不需要很新颖的设计，慢慢学习ground truth就能够给出较好的结果，同时由于box regression的存在，能够对候选框进行修正

思路简单：
0.给图像，跑CNN（论文中使用了ZF 和 VGG-16），选择需要提取特征的feature map
1.将该feature map扔到RPN计算候选区域
2.得到候选区域后，拿着feature map 和 候选区域 进行和fast R-CNN一样的计算
![](https://upload-images.jianshu.io/upload_images/9165719-a1e6666e68709f55.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

简单的总结就是
0.Faster R-CNN使用RPN进行候选框的生成，基于神经网络训练时间长但是计算时间短的特性，在RPN网络训练好后，生成的候选区域无论是质量还是速度都是碾压selective search的
1.但是根据faster R-CNN的结构我们很容易就看出，Faster R-CNN训练时间要长于Fast R-CNN（这也就解释了为什么下图作者很鸡贼地没有提到训练时间加速的问题2333）
2.faster R-CNN使得实时检测变得可行

![](https://upload-images.jianshu.io/upload_images/9165719-8832851f1a0d5c5c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

####Mask R-CNN
相较于上述网络相比，Mask R-CNN可以说是非常惊艳了
以上的网络都有一个固定的架构模板，Mask R-CNN却是灵活多变的，它更像是一种网络架构的方法而不是既定的网络
它在人体姿态检测和物体分割等等任务中的表现出乎意料得好，拿下了ICCV 2017最佳论文

mask R-CNN建立在fast R-CNN 和 faster R-CNN上，
第一阶段是RPN生成候选区域，第二阶段在分类与候选区域回归的基础上加入了并行的用于预测mask的分支
该分支对任意大小的RoI输入，输出Km<sup>2</sup>大小的masks，K对应K类分类，m x m 是 RoI放缩后的大小，输出的内容是，对于K类分类中的第i张m x m，1表示该点属于第i类，0表示该点不属于i th class

一些重要的变化就是
0.使用了RoIAlign替换了RoI Pool，因为RoI Pool 存在的问题是，对于输入类似于M x N的图像，要求输出是固定的a x b， M / a 或 N / b不一定是整数，这就导致需要对坐标进行取整，而在maxpooling后，生成的feature map与原RoI会有一定偏差，虽然在Faster R-CNN中这一误差影响不大，但在像素级mask计算时，这个误差会导致计算的mask与原图对应位置产生偏移（分类框偏移本身不会对分类造成太大影响，但mask偏移可能原图中对应的分割就会少了一个指甲盖，多了一部分背景）。RoIAlign使用了类似的
1.加入mask branch，与box regression 和 classification 同时运算（与其他相似工作中，先计算mask后利用mask分类不同）

下图为Mask R-CNN与 faster R-CNN的区别
![](https://upload-images.jianshu.io/upload_images/9165719-056cf1d32a7c4703.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

而之所以说mask R-CNN非常惊艳，是因为他不仅仅能做图像分割，物体识别定位，还能够做人体姿态识别

![](https://upload-images.jianshu.io/upload_images/9165719-dd2881494836286a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

而做人体姿态识别对于mask R-CNN而言也是很简单的：
将上述mask R-CNN网络的K类分类，改为人体的K个关键点：如左肩，左手，眉心，双眼，脖子等等
在生成mask的时候mask R-CNN会计算出人体的K个关键点，将这些关键点连接起来就是人体姿态

实现代码网络上已经有很多了，我大概会在国庆结束前或者永远都不会给出pytorch 或者 tensorflow的代码（逃
