from __future__ import division #把下一版本的特性导入当前版本中

import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import *

'''
我们可以将一个假的层置于之前提出的路由层的位置上，然后直接在代表 darknet 的 nn.Module 对象的 forward 函数中执行拼接运算
'''
class EmptyLayer(nn.Module):
'''
对于在 Route 模块中设计一个层，我们必须建立一个 nn.Module 对象，其作为 layers 的成员被初始化。然后，我们可以写下代码，将 forward 函数中的特征图拼接起来并向前馈送。最后，我们执行网络的某个 forward 函数的这个层
'''
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
            


        
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416)) #Resize to the input dimension
    img_ = img[:,:,::-1].transpose((2,0,1)) # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0 #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float() #Convert to float
    img_ = Variable(img_) # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    Takes a configuration file
 
    返回一个blocks的list,每一个blcks描述了神经网络一个block,block在list中是字典
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n') # store the lines in a list
    lines = [x for x in lines if len(x) > 0] # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#'] # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces

    block = {}#是字典，里面是key-value,每个block就是一层
    blocks = [] #是一个数组存放block
    
    for line in lines:
        if line[0] == "[": # This marks the start of a new block
            if len(block) != 0: # If block is not empty, implies it is storing values of previous block.
                 blocks.append(block) # add it the blocks list
                 block = {} # re-init the block
            block["type"] = line[1:-1].rstrip() #其中1是第二个字符，而-1是倒数一个字符所以取得[]里的字符
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

def create_modules(blocks):
    net_info = blocks[0] #Captures the information about the input and pre-processing 
    module_list = nn.ModuleList()#返回一个包含nn.Module对象的普通列表
    '''
     当我们定义一个新的卷积层时，我们必须定义它的卷积核维度。虽然卷积核的高度和宽度由 cfg 文件提供，但卷积核的深度是由上一层的卷积核数量（或特征图深度）决定的。这意味着我们需要持续追踪被应用卷积层的卷积核数量。我们使用变量 prev_filter 来做这件事。我们将其初始化为 3，因为图像有对应 RGB 通道的 3 个通道
    '''
    '''
     路由层（route layer）从前面层得到特征图（可能是拼接的）。如果在路由层之后有一个卷积层，那么卷积核将被应用到前面层的特征图上，精确来说是路由层得到的特征图。因此，我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上
    '''
    prev_filters = 3 #初始时,因为图片有RGB三个通道，所以prev_filters为3有3个核
    output_filters = [] #每个模块的输出卷积核数量在output_filters
   # filters = prev_filters
    
    for index, x in enumerate(blocks[1:]):
        #filters = prev_filters
        module = nn.Sequential() #每个模块可能很多层,用nn.Sequential将各层串起来
        #check the type of block
        #create a new module for the block
        #append to module_list
        if (x["type"] == "convolutional"):
         #Get the info about the layer
            activation = x["activation"]
            try:
                 batch_normalize = int(x["batch_normalize"]) #0 1代表是否要批归一化
                 bias = False
            except:
                 batch_normalize = 0
                 bias = True
            filters= int(x["filters"]) #卷积核的数量
            padding = int(x["pad"]) # 0 1 代表是否pad
            kernel_size = int(x["size"]) #卷积核的大小
            stride = int(x["stride"]) #步长
        
            if padding:
                 pad = (kernel_size - 1) // 2 # //整除
            else:
                 pad = 0
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv) #加入卷积模块
        
            if batch_normalize:
                 bn = nn.BatchNorm2d(filters)
                 module.add_module("batch_norm_{0}".format(index), bn) #加入批归一化模块
            if activation == "leaky":
                 activn = nn.LeakyReLU(0.1, inplace = True)
                 module.add_module("leaky_{0}".format(index), activn) #加入激活函数
        elif (x["type"] == "upsample"):
             stride = int(x["stride"])
             '''
             如果mode是 bilinear会报错 说什么没有filters？？？？？？？？？？？？
             '''
             upsample = nn.Upsample(scale_factor = 2, mode = "nearest")#双线性上采样，其实就是入大
             module.add_module("upsample_{}".format(index), upsample)
             
        elif (x["type"] == "route"):
             x["layers"] = x["layers"].split(',')
             #Start of a route
             start = int(x["layers"][0])
             try:
                 end = int(x["layers"][1])#有的有两个参数
             except:
                 end = 0
            #Positive anotation
             if start > 0: 
                 start = start - index #得到相对于index的相对顺序
             if end > 0:
                 end = end - index    #得到相对于index的相对顺序
             route = EmptyLayer() #会在forward中执行并接
             module.add_module("route_{0}".format(index), route)
             if end < 0:
                 filters = output_filters[index + start] + output_filters[index + end]
             else:
                 filters= output_filters[index + start]
        elif (x["type"] == "shortcut"):
             shortcut = EmptyLayer()
             module.add_module("shortcut_{}".format(index), shortcut)
                 
             '''
             还不清楚
             '''
        elif (x["type"] == "yolo"):
             mask = x["mask"].split(",") #'0 1 2'
             mask = [int(x) for x in mask] #[0 1 2]
             
             '''
             配置文件中有3组yolo层，所以每层的mask从0-8,第一层用anochor 0-3 以此类推
             '''
             anchors = x["anchors"].split(",")
             anchors = [int(a) for a in anchors]#' 10 13 16 30 33 23 ...'
             anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]#(10 13),(16,30),(33,23)...
             anchors = [anchors[i] for i in mask]#[(),(),()]

             detection = DetectionLayer(anchors)#DetectionLayer对像中anchors中保存了anchors
             module.add_module("Detection_{}".format(index), detection)
            
                
        module_list.append(module)
        prev_filters = filters#有的层没有filters怎么加呢(上面的else if??????????????????????????????????????????
        output_filters.append(filters)
    return (net_info, module_list)

'''
def create_modules2(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
    
        #check the type of block
        #create a new module for the block
        #append to module_list
        
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)
                
        #If it is a route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net_info, module_list)
'''
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        
    def forward(self, x, CUDA = False):
        modules = self.blocks[1:]
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    #我们在字典 outputs 中缓存每个层的输出特征图。关键在于层的索引，且值对应特征图。
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == 'yolo': 
            '''
            YOLO 的输出是一个卷积特征图，包含沿特征图深度的边界框属性。边界框属性由彼此堆叠的单元格预测得出。因此，如果你需要在 (5,6) 处访问单元格的第二个边框，那么你需要通过 map[5,6, (5+C): 2*(5+C)] 将其编入索引。这种格式对于输出处理过程（例如通过目标置信度进行阈值处理、添加对中心的网格偏移、应用锚点等）很不方便。

另一个问题是由于检测是在三个尺度上进行的，预测图的维度将是不同的。虽然三个特征图的维度不同，但对它们执行的输出处理过程是相似的。如果能在单个张量而不是三个单独张量上执行这些运算，就太好了
            '''
                anchors = self.module_list[i][0].anchors
                #Get the input dimensions
                inp_dim = int (self.net_info["height"])
        
                #Get the number of classes
                num_classes = int (module["classes"])
        
                #Transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              #if no collector has been intialised. 
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
            outputs[i] = x #在这缓冲每层的输出
        
        return detections
    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)