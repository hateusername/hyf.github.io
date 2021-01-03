![image.png](http://upload-images.jianshu.io/upload_images/9165719-77c5b6abda9fc17c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[文章下载请戳此处](https://arxiv.org/abs/1508.06576)

####这是我读的deep learning第一篇paper

当初巨佬说如果对深度学习感兴趣，可以先看点科普读物然后直接看ICCV，当时我还有点将信将疑，说自己没有这个能力还是希望能多多积累一些知识，读完这篇paper后发现其实DL这个领域并不是水，而是paper就像一篇小说，大家都能读哈利波特，但不是谁都能写。DL的paper确实好读，如果有更充足的准备以及更好的预备知识，积累了一定项目经验后，一天读十几篇paper并不是玩笑话。

不过这也是一家之言了，毕竟我现在只读过一篇paper XD，记录一下心路历程等以后变微佬回来看看自己当年有多撒币hhh

2018年中更新↑这个人是傻逼不要理他↑

#内容提要
艺术 = 内容 + 风格
内容：深层卷积探测到的feature map
风格：各个feature map的内积

2018年中更新↑emm这个人比较菜请不要过多参考

#实现
##CONTENT_ IMAGE
![tj.jpg](https://upload-images.jianshu.io/upload_images/9165719-9ff0dc017f57749f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
##STYLE_IMAGE
![vc.jpg](https://upload-images.jianshu.io/upload_images/9165719-69821e066d924e0c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
##RESULT_IMAGE
#####迭代1000次
![output_1000.jpg](https://upload-images.jianshu.io/upload_images/9165719-36a13b76c738c771.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#####迭代5000次
![output_5000.jpg](https://upload-images.jianshu.io/upload_images/9165719-f732fcaded76befd.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#####迭代10000次
![final.jpg](https://upload-images.jianshu.io/upload_images/9165719-110e513980afe32b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#参考资料
1. 论文
2. VGG19的数据结构与网络结构，如何取得自己想要的数据（从.mat文件中取出），以及搭建一个VGG
3. github上有很多这个东西的实现，我写完了懒得思考了再见2333

动手这种东西，还是自己来吧，我也说不出我学到了什么，反正我做出来了
扶额.JPG

#代码
参考资料：https://github.com/ckmarkoh/neuralart_tensorflow
我的代码：
```
import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
import os

#以后可以reshape 图片，在scipy.misc doc里有
IMAGE_W = 800
IMAGE_H = 600
TIME_iter = 10000
MEAN_VALUES = np.array([123,117,104]).reshape((1,1,1,3))
RATIO = 500 # LOSS = αLc + βLs  RATIO = α/β
INIT_NOISE_RATIO = 0.7

path_content = './images/tj.jpg'
path_texture = './images/vc.jpg'
path_VGG = './VGG19.mat'
path_out_father = './output/'
name_out = 'output.jpg'


#CONTENT_LAYERS = 'conv4_2'
#TEXTURE_LAYERS = [conv1_1,conv2_1,conv3_1,conv4_1,conv5_1]


def get_wb(vgg_layers,num_layer):
    weight = vgg_layers[num_layer][0][0][0][0][0]
    # VGG19.mat 文件数据结构
    weight = tf.constant(weight)
    bias = vgg_layers[num_layer][0][0][0][0][1]
    bias = tf.constant(np.reshape(bias,(bias.size)))
    return weight,bias

def add_net(net_type,net_from,weight_bias=None):
    if net_type == 'conv':
        #conv 与 relu 层 直接写在一起
        return tf.nn.relu(tf.nn.conv2d(net_from,weight_bias[0],[1,1,1,1],'SAME') + weight_bias[1])
    elif net_type =='pool':
        return tf.nn.avg_pool(net_from,[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def read_img(path):
    img = scipy.misc.imread(path)
    img = scipy.misc.imresize(img,(IMAGE_H,IMAGE_W))
    img = img[np.newaxis,:,:,:]
    img = img - MEAN_VALUES
    return img

def write_img(img,path_out,name_out):
    img = img + MEAN_VALUES
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    scipy.misc.imsave(path_out + name_out +'.jpg', img)

def content_loss(p,x):
    #squared error of conv4_2
    M = p.shape[1] * p.shape[2] #VGG的数据结构
    N = p.shape[3]
    loss = (1. / (2 * N ** 0.5 * M **0.5)) * tf.reduce_sum(tf.pow((x - p),2))
    return loss

def Gram_Matrix(layer,M,N):
    x1 = tf.reshape(layer, (M, N))
    g = tf.matmul(tf.transpose(x1), x1)
    return g


def texture_loss(p,x):
    #需要矩阵
    N = p.shape[3]
    M = p.shape[1]*p.shape[2]
    G = Gram_Matrix(p,M,N)
    A = Gram_Matrix(x,M,N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def main():
    '''
        layers = (
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4','pool3',
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
            )
        '''
    vgg_net = scipy.io.loadmat(path_VGG)
    vgg_layers = vgg_net['layers'][0]  # 参考VGG.mat 文件结构

    input = tf.Variable(np.zeros((1, IMAGE_H, IMAGE_W, 3)).astype('float32'))

    conv1_1 = add_net('conv', input, get_wb(vgg_layers, 0))
    conv1_2 = add_net('conv', conv1_1, get_wb(vgg_layers, 2))
    pool1 = add_net('pool', conv1_2)

    conv2_1 = add_net('conv', pool1, get_wb(vgg_layers, 5))
    conv2_2 = add_net('conv', conv2_1, get_wb(vgg_layers, 7))
    pool2 = add_net('pool', conv2_2)

    conv3_1 = add_net('conv', pool2, get_wb(vgg_layers, 10))
    conv3_2 = add_net('conv', conv3_1, get_wb(vgg_layers, 12))
    conv3_3 = add_net('conv', conv3_2, get_wb(vgg_layers, 14))
    conv3_4 = add_net('conv', conv3_3, get_wb(vgg_layers, 16))
    pool3 = add_net('pool', conv3_4)

    conv4_1 = add_net('conv', pool3, get_wb(vgg_layers, 19))
    conv4_2 = add_net('conv', conv4_1, get_wb(vgg_layers, 21))
    conv4_3 = add_net('conv', conv4_2, get_wb(vgg_layers, 23))
    conv4_4 = add_net('conv', conv4_3, get_wb(vgg_layers, 25))
    pool4 = add_net('pool', conv4_4)

    conv5_1 = add_net('conv', pool4, get_wb(vgg_layers, 28))
    conv5_2 = add_net('conv', conv5_1, get_wb(vgg_layers, 30))
    conv5_3 = add_net('conv', conv5_2, get_wb(vgg_layers, 32))
    conv5_4 = add_net('conv', conv5_3, get_wb(vgg_layers, 34))
    pool5 = add_net('pool', conv5_4)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    img_noise = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
    img_content = read_img(path_content)
    img_texture = read_img(path_texture)

    sess.run(input.assign(img_content))
    loss_content = content_loss(sess.run(conv4_2),conv4_2)

    texture_layers = [conv1_1,conv2_1,conv3_1,conv4_1,conv5_1]
    sess.run(input.assign(img_texture))
    loss_texture =(1/5) * sum(map(lambda layer: texture_loss(sess.run(layer),layer),texture_layers))

    loss_total = loss_content + RATIO * loss_texture
    optimizer = tf.train.AdamOptimizer(2.0)

    train = optimizer.minimize(loss_total)
    sess.run(tf.initialize_all_variables())
    sess.run(input.assign(INIT_NOISE_RATIO * img_noise + (1.-INIT_NOISE_RATIO) * img_content))

    for i in range(TIME_iter):
        sess.run(train)
        if i % 100 == 0:
            out_img = sess.run(input)
            print("Iterated   ",i,"   times")
            sess.run(loss_total)
            write_img(out_img,path_out_father,'output_'+ str(i))
    final_img = sess.run(input)
    write_img(final_img,path_out_father,'final')


'''         读取图片
img_cont = scipy.misc.imread(content)
print(type(img_cont))
print(img_cont.dtype)
print(img_cont.shape)


      输出一个速随机的长800高600的图片至 输出目录
x = np.random.random((600,800,3))
scipy.misc.imsave(out_father_dir + 'test.jpg',x)
'''

if __name__ == '__main__':
    main()
```

