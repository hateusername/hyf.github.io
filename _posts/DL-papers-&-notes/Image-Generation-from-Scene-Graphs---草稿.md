[下载链接
__Image Generation from Scene Graphs__](https://arxiv.org/abs/1804.01622)


##以下为草稿！
##阅读源代码后将给出详细工作细节

简要概括
输入句子，输出符合句子描述的图像

1.将句子转为场景图（人工）
2.场景图被图卷积网络处理成合适的向量
3.生成的向量用于计算掩膜(masks)以及bounding boxes
4.级联精化网络(或者细化？cascaded refinement network)将模糊的masks精细化放置在bounding box内。至此图像生成结束，以上为G
5.D<sub>obj</sub>保证图片中物体真实可辨认，D<sub>img</sub>保证图片整体真实

GD对抗训练，同时最小化6个loss加权和:
bounding box罚函数，mask罚函数，pixel罚函数，D<sub>img</sub>对抗loss，D<sub>obj</sub>对抗loss，D<sub>obj</sub>obj识别loss

单卡P100训练用时3天
