本文讨论how VGG work，而不是 why VGG work

即：VGG中数据传递的方式，输入一张图片，产生什么数据，如何与下一层对接
并非：VGG为什么能理解图片内容（这是玄学）

本文适合对卷积网络有大致了解，但是不清楚具体工作细节的读者
如果完全没听过卷积网络请先大概有个了解

###请先阅读以下材料，全部理解清楚，稍微思考就能完全理解VGG了
#####VGG的结构（重要）
参考网站：http://cs231n.github.io/convolutional-networks/
下图表示的是vgg的结构
![image.png](https://upload-images.jianshu.io/upload_images/9165719-1d43e97999efc1e0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#####卷积的运算方式（重要）
下图表示的是卷积中的具体运算方法（参考网站中，这张图是动态的，点击toggle movement能够暂停）
![image.png](https://upload-images.jianshu.io/upload_images/9165719-d4bf22e3440c135d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#####VGG参数需要占用的空间（极大帮助理解VGG）
```
INPUT: [224x224x3]        memory:  224*224*3=150K   weights: 0
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*3)*64 = 1,728
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*64)*64 = 36,864
POOL2: [112x112x64]  memory:  112*112*64=800K   weights: 0
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*128)*128 = 147,456
POOL2: [56x56x128]  memory:  56*56*128=400K   weights: 0
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2: [28x28x256]  memory:  28*28*256=200K   weights: 0
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2: [14x14x512]  memory:  14*14*512=100K   weights: 0
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2: [7x7x512]  memory:  7*7*512=25K  weights: 0
FC: [1x1x4096]  memory:  4096  weights: 7*7*512*4096 = 102,760,448
FC: [1x1x4096]  memory:  4096  weights: 4096*4096 = 16,777,216
FC: [1x1x1000]  memory:  1000 weights: 4096*1000 = 4,096,000

TOTAL memory: 24M * 4 bytes ~= 93MB / image (only forward! ~*2 for bwd)
TOTAL params: 138M parameters
```
#####不同VGG的实现细节（辅助阅读不想看可以跳过）
下图是VGG的论文中给出的VGG具体结构
![image.png](https://upload-images.jianshu.io/upload_images/9165719-4c9281620923cb5b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#####VGG细节（辅助阅读）
下图是VGG论文中给出的数据
![image.png](https://upload-images.jianshu.io/upload_images/9165719-6a3271c04f45495f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
所有的convolution stride都为1，filter都是 3x3大小，padding使得图像大小不变，max pooling 2x2,stride 2


####结合以上资料，给出VGG的工作细节(请先阅读参考网站，并搞清楚所有重要内容)
#####输入层（input layer）
224x224x3 即图像宽高为224，剩下的三个通道对应RGB
#####第一层卷积层，conv1_1:
由第一张图，第一层卷积层参数为224x224x64 代表第一层卷积层有224x224x64个神经元，其中，每224x224为一小层神经元，这一小层神经元参数共享，所以这224*224的神经元参数完全相同，共同形成了一个filter，每一个神经元的维度都是3,3,3 分别对应filter_hight,filter_width,channels（即RGB通道） ，所以总共有64个filter（其实总共有224x224x64个filter但是由于参数共享，每层神经元参数是一样的，所以可以理解为形成了一个单独的在整张图上移动的filter），所以64个filter的参数每个都是 3x3x3

第二张图中给出了卷积的具体运算方式，可知，每一个3X3X3的filter对一张224X224X3的图片进行一次读取计算操作之后（不移动），得到一个具体的数值，存储于对应的feature map中，移动一小步后再次进行读取计算产生下一个值，扫描一次完整的图片，每个filter 能够生成224x224大小的feature map。224x224x3的图片在经过64个filter的第一层后，生成的数据应该是，224x224x64。所以第一层的weight参数数量应该是(filter_hight x filter_width x channels) x 64（个filter）即 3x3x3x64

#####第二层卷积层，conv1_2:
该层仍为224x224x64个神经元，64个filter，每个filter都是3x3x64（上一层为3x3x3因为输入图像是224x224x3的，但是第二层输入数据是224x224x64的），这也就解释了，第二层weight参数数量为（3x3x64）x64

按照这个方式，所有参考文档都能够契合在一起，也就理解了VGG基本的工作结构
