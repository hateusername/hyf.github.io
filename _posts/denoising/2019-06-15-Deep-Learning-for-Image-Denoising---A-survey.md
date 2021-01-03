传统方法弊端：
1.速度慢
2.非凸，需要手动优化

normalized initialization可以阻止gradient vanishing or exploding
GAN也被用来降噪
deep learning 在降噪领域目前没有成熟的理论

blind image denoising
BN Relu能够加速

数据集BSD68（

限制
1.目前限制于AWGN,不能处理real image，like low light image
2.没有一体模型 = denoising + super resolution + blurring + deblocking
3.cant use a model to address blind gaussian noise
