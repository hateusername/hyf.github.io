基本思路：
使用resnet101 + FPN + RPN 作为RPN网络用于生成RoI

######pytorch中调用预训练的resnet101：
```original_resnet = models.resnet101(pretrained=True)```
```pretrained = True会下载预训练的参数，False仅下载网络结构```


```print(list(original_resnet.children()))```能够观察网络内部的结构
[输出如下](https://www.jianshu.com/p/8c2f6353e1eb)


```print(original_resnet._modules.keys())```可以看见网络结构关键词
```odict_keys(['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])```

我们知道conv层对输入是没有要求的，但是FC对输入会有严格限制，由于我们不需要最终的分类信息，所以我们将fc层删去
```resnet = nn.Sequential(*list(original_resnet.children())[:-1])```
新的resnet关键词信息与原先original不同，但是没有大碍，我们仍然使用keys（）可以得到

######取出conv层的信息：
pytorch会忽略中间层数据，计算完后直接扔掉（官方解释是为了节约空间）
1.如果只需要取出单层信息，[这里](https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3)有一个简单的方法，将需要的信息作为最后一层直接输出
2.我们搭建的FPN需要四个conv层的全部feature maps，为了获取中间CONV层的信息我们需要使用hook工具将计算过程中生成的数据提前取出来存储，避免在后期计算遗失

记录第一层maxpool后的信息
```h1 = resnet[3].register_forward_hook(hook)```

我们首先进行一次前向
```resnet(input)``` 此处的input是一张图片转换成的tensor，由于删去了fc层你可以使用任意的输入尺寸，但是注意，在pytorch中conv2d层的输入要求是![](https://upload-images.jianshu.io/upload_images/9165719-b5d6ada106ddedce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
应当注意匹配，上图对应为batch，channel（RGB），hight，width

前向完成后，信息储存在自己定义的hook中，我们将64个feature map 全部导出为图片可视化
放两张意思一下
![](https://upload-images.jianshu.io/upload_images/9165719-74c15e51112a4ff7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/9165719-9f1c8401ca6475b7.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/9165719-8fc88cbcccdfc1bb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

那么到此为止我们就能够顺利取出feature maps了，接下来的任务就是利用feature maps 搭建FPN + RPN

