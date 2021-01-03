阅读顺序GAN,DCGAN,WGAN,WGAN-GP,LSGAN,BEGAN等等（看完了再更新）
论文下载：
[GAN](https://arxiv.org/abs/1406.2661)
[DCGAN](http://arxiv.org/abs/1511.06434)
[WGAN](https://arxiv.org/abs/1701.07875)
[WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf)
[LSGAN](https://arxiv.org/pdf/1611.04076.pdf)

#####GAN Ian Goodfellow

G（ z , θ<sub>g</sub> ）=  参数θ<sub>g</sub>控制的生成器，输入为噪声z，输出为伪样本
D（ x , θ<sub>d</sub> ）=  参数θ<sub>d</sub>控制的判别器，输入为样本x（数据集中的，或者是G生成的），输出为输入样本属于数据集（不属于G生成）的概率

训练：
D：提高D判别的正确率
G：提高生成的样本通过D判别为数据集的概率（欺骗D）,1 - D(G(z))

即为![](https://upload-images.jianshu.io/upload_images/9165719-c16fe2e5e7b15216.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
式中第一项为，x∈给定数据集，提高D(x)
第二项为，x∉数据集，给定随机输入噪声z，提高 1 - D(G(z)),其中G(z)为伪样本，D(G(z))为伪样本经过D判定属于数据集的概率，1 - D(G(z))为D判定不属于数据集的概率，降低该概率

理论部分：
1.SGD训练GAN算法
2.存在全局最优解，当G分布 = 数据分布时成立
3.理论1算法收敛
详细内容见论文

#####DCGAN deep convoluntional GAN
论文主要工作：
1.提出了能够使得DCGAN训练稳定的拓扑要求
2.使用了图像分类中成熟的分类器作为discriminator
3.可视化卷积层中的filter发现了其工作效果
4.G存在向量计算特性

核心为DCGAN稳定训练的要求：
![](https://upload-images.jianshu.io/upload_images/9165719-7df41011ef524135.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可视化部分文字总结会限制你们的思维，还是你们自己看论文去发现特点吧

矢量特性附论文截图
![](https://upload-images.jianshu.io/upload_images/9165719-0d10419aa6ff3961.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#####WGAN

####自己的笔记（自己存着怕丢了，不是给你们看的⁄(⁄ ⁄•⁄ω⁄•⁄ ⁄)⁄）
![](https://upload-images.jianshu.io/upload_images/9165719-1128973f4b6a6df8.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)

![](https://upload-images.jianshu.io/upload_images/9165719-fef883120fa338e9.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)

![](https://upload-images.jianshu.io/upload_images/9165719-0ff834b592c9ede3.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)
