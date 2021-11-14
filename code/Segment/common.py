import tensorflow as tf

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

class Conv2D(tf.keras.layers.Conv2D):
    def __init__(self,f,k=3,a='mish',b=True,s=1,p='same',bn=True,**kwdarg):
        if a == 'mish':
            a = None
            self.mish = True
        else:
            self.mish = False
        if bn:
            b = False
            self.has_bn = True
            self.bn = tf.keras.layers.BatchNormalization()
        else:
            self.has_bn = False
        super().__init__(
            filters=f,
            kernel_size=k,
            activation=a,
            padding=p,
            use_bias=b,
            **kwdarg)
    def call(self,inp,training):
        x = super().call(inp)
        if self.has_bn:
            x = self.bn(x,training=training)
        if self.mish:
            x = mish(x)
        return x

class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self,m=1,k=3,a='mish',b=True,s=1,p='same',bn=True,**kwdarg):
        if a == 'mish':
            a = None
            self.mish = True
        else:
            self.mish = False
        if bn:
            b = False
            self.has_bn = True
            self.bn = tf.keras.layers.BatchNormalization()
        else:
            self.has_bn = False
        super().__init__(
            depth_multiplier=m,
            kernel_size=k,
            activation=a,
            padding=p,
            use_bias=b,
            **kwdarg)
    def call(self,inp,training):
        x = super().call(inp)
        if self.has_bn:
            x = self.bn(x,training=training)
        if self.mish:
            x = mish(x)
        return x

class DepthControl(tf.keras.layers.Layer):
    def __init__(self,r,a=None):
        super().__init__()
        self.r = 2**r
        self.a = a
    def build(self,input_shape):
        self.conv = Conv2D(
            f=int(input_shape[-1]*self.r),
            k=1,
            a=self.a,
            b=False,
            s=1,
            p='same',
            bn=False)
    def call(self,inp,training):
        x = self.conv(inp,training=training)
        return x

class DownSample(tf.keras.layers.Layer):
    def __init__(self,r=1):
        super().__init__()
        self.r = 2**r
    def build(self,input_shape):
        self.kernel = tf.fill([self.r, self.r, input_shape[-1], 1],1/self.r**2)
    def call(self,inp,training):
        return tf.nn.depthwise_conv2d(inp,self.kernel,strides=[1,self.r,self.r,1],padding='SAME')

class UpSample(tf.keras.layers.UpSampling2D):
    def __init__(self,r=1):
        super().__init__(2**r,interpolation='bilinear')
    def call(self,inp,training):
        return super().call(inp)

class Contract(tf.keras.layers.Layer):
    def __init__(self,a=None):
        super().__init__()
        self.downsample = DownSample()
        self.increase = DepthControl(r=1,a=a)
    def call(self,inp,training):
        x = self.downsample(inp,training=training)
        x = self.increase(x,training=training)
        return x

class Expend(tf.keras.layers.Layer):
    def __init__(self,a=None):
        super().__init__()
        self.upsample = UpSample()
        self.decrease = DepthControl(r=-1,a=a)
    def call(self,inp,training):
        x = self.upsample(inp,training=training)
        x = self.decrease(x,training=training)
        return x

class VisionField(tf.keras.layers.Layer):
    def __init__(self,k=3,a='mish'):
        super().__init__()
        self.k = k
        self.a = a
    def build(self,input_shape):
        self.conv = Conv2D(
            f=input_shape[-1],
            k=self.k,
            a=self.a,
            b=False,
            s=1,
            p='same',
            bn=True)
    def call(self,inp,training):
        x = self.conv(inp,training=training)
        return x

class DepthwiseVisionField(tf.keras.layers.Layer):
    def __init__(self,k=3,a='mish'):
        super().__init__()
        self.dconv = DepthwiseConv2D(
            m=1,
            k=k,
            a=a,
            b=False,
            s=1,
            p='same',
            bn=True)
    def call(self,inp,training):
        x = self.dconv(inp,training=training)
        return x

class SubBlock(tf.keras.layers.Layer):
    def __init__(self,r=-1):
        super().__init__()
        self.decrease = DepthControl(r=r,a='mish')
        self.depthwise_vf = DepthwiseVisionField()
        self.increase = DepthControl(r=-r,a='mish')
    def call(self,inp,training):
        x = self.decrease(inp,training=training)
        x = self.depthwise_vf(x,training=training)
        x = self.increase(x,training=training)
        x = x + inp
        return x

class SequentialSubBlock(tf.keras.layers.Layer):
    def __init__(self,repeat,r=-1):
        super().__init__()
        self.subblocks = [SubBlock(r=r) for _ in range(repeat)]
    def call(self,inp,training):
        x = inp
        for sb in self.subblocks:
            x = sb(x,training=training)
#         x = x + inp
        return x

class ResBlock(tf.keras.layers.Layer):
    def __init__(self,repeat,r=-1):
        super().__init__()
        self.subblocks = SequentialSubBlock(repeat=repeat,r=r)
        self.res_conv = DepthControl(r=0,a='mish')
        self.x_conv1 = DepthControl(r=0,a='mish')
        self.x_conv2 = DepthControl(r=0,a='mish')
        self.mix = DepthControl(r=-1,a='mish')
    def call(self,inp,training):
        res = self.res_conv(inp,training=training)
        x = self.x_conv1(inp,training=training)
        x = self.subblocks(x,training=training)
        x = self.x_conv2(x,training=training)
        x = tf.concat([x,res],axis=-1)
        x = self.mix(x,training=training)
        return x

class ResBlockContract(tf.keras.layers.Layer):
    def __init__(self,repeat,r=-1):
        super().__init__()
        self.resblock = ResBlock(repeat,r=r)
        self.contract = Contract()
    def call(self,x,training):
        x = self.resblock(x,training=training)
        x = self.contract(x,training=training)
        return x

