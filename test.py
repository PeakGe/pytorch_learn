import torch
import torch.nn as nn
import numpy as np

# weekday=[('Monday','周一'),('Tuesday','周二'),('Wednesday','周三'),('Thursday','周四'),('Friday','周五')]
# for step,(w1,w2) in enumerate(weekday):
#     print(step,w1,w2)
# label=1
# desc=f"Computing AP for class '{label}'"
# print(desc)
#
# x=torch.linspace(-1,1,100)
# print(x.size())
#
# y0 = torch.zeros(100)
# print(y0.size())
class DetectionLayer(nn.Module):	#检测层
    def __init__(self, anchors):	#初始化参数:anchors
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

module_list = nn.ModuleList()
module=nn.Sequential()

mask = "6,7,8".split(",")
mask = [int(x) for x in mask]

# 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326".split(",")
anchors = [int(a) for a in anchors]
anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
# anchors[i]为一个2元组
anchors = [anchors[i] for i in mask]

# 定义1个检测层:参数:anchors
detection = DetectionLayer(anchors)
module.add_module("Detection_{}".format(0), detection)
module.add_module("Detection_{}".format(1), detection)

# 每得到1个module就放入module_list之中
module_list.append(module)

xx=module_list[0]
print(xx)
anchors = xx[1].anchors
print(anchors)

grid = np.arange(13)
a,b = np.meshgrid(grid, grid)

print(torch.FloatTensor(a).size())
print(torch.FloatTensor(b).size())
x_offset = torch.FloatTensor(a).view(-1,1)
print(x_offset.size())
y_offset = torch.FloatTensor(b).view(-1,1)
print(y_offset.size())

x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,3).view(-1,2).unsqueeze(0)
print(x_y_offset.size())

anchors= [(116, 90), (156, 198), (373, 326)]
anchors = [(c[0]/32, c[1]/32) for c in anchors]
print(type(anchors))
anchors = torch.FloatTensor(anchors)
anchors = anchors.repeat(13*13, 1).unsqueeze(0)
print(type(anchors))
print(anchors.size())
