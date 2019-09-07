# [MobileNet](https://github.com/tfzoo/MobileNet) 
## [MobileNet简介](https://github.com/tfzoo/MobileNet/wiki) 

MobileNet是为移动和嵌入式设备提出的高效模型。MobileNets基于流线型架构(streamlined)，放弃pooling直接采用stride = 2进行卷积运算，使用深度可分离卷积(depthwise separable convolutions,即Xception变体结构)来构建轻量级深度神经网络。

基本单元深度级可分离卷积（depthwise separable convolution）这种结构之前已经被使用在Inception模型中，其实是一种可分解卷积操作（factorized convolutions），其可以分解为两个更小的操作：DW（depthwise convolution） 和 PW（pointwise convolution）；depthwise convolution是depth级别的操作，而pointwise convolution其实是采用1x1的卷积核的普通卷积。

mobilenetv2 与mobilenetV1 不同点：

* 1、引入了shortcut结构（残差网络）
* 2、在进行depthwise之前先进行1x1的卷积增加feature map的通道数，实现feature maps的扩张。
* 3、pointwise结束之后弃用relu激活函数，改用linear激活函数，来防止relu对特征的破坏。

```
# PP DEMO

class MobileNet():
    def __init__(self):
        pass
    
    def name(self):
        return 'mobile-net'

    def net(self, input, class_dim=1000, scale=1.0):
        # conv1: 112x112
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)

        # 56x56
        input = self.depthwise_separable(
            input,
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale)

        input = self.depthwise_separable(
            input,
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=2,
            scale=scale)

        # 28x28
        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale)

        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=2,
            scale=scale)

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale)

        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=2,
            scale=scale)

        # 14x14
        for i in range(5):
            input = self.depthwise_separable(
                input,
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                scale=scale)
        module1 = input
        # 7x7
        input = self.depthwise_separable(
            input,
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=2,
            scale=scale)

        input = self.depthwise_separable(
            input,
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=1,
            scale=scale)

        # class_dim x 1
        input = paddle.fluid.layers.conv2d(
            input,
            num_filters=class_dim,
            filter_size=1,
            stride=1)

        pool = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_stride=1,
            pool_type='avg',
            global_pooling=True)

        output = fluid.layers.fc(input=pool,
                              size=class_dim,
                              act='softmax', 
                              param_attr=ParamAttr(initializer=MSRA()))
        
        return output

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      act='relu',
                      use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(initializer=MSRA()),
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def depthwise_separable(self, input, num_filters1, num_filters2, num_groups,
                            stride, scale):
        depthwise_conv = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=True)

        pointwise_conv = self.conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)
        return pointwise_conv

```

###  [天府动物园 tfzoo：tensorflow models zoo](http://www.tfzoo.com)
####   qitas@qitas.cn
[![sites](tfzoo/tfzoo.png)](http://www.tfzoo.com)