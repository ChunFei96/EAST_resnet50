import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

import resnet

class extractor(nn.Module):
    def __init__(self, pretrained):
        super(extractor, self).__init__()

        self.backbone = resnet.resnet50()
	
    def forward(self, x):
        out =[]
        x2, x3, x4, x5 = self.backbone(x)
        out.append(x2)
        out.append(x3)
        out.append(x4)
        out.append(x5)

        return out


class merge(nn.Module):
	def __init__(self):
		super(merge, self).__init__()

		#self.conv1 = nn.Conv2d(1024, 128, 1)
		self.conv1 = nn.Conv2d(3072, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		#self.conv3 = nn.Conv2d(384, 64, 1)
		self.conv3 = nn.Conv2d(640, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		# self.conv5 = nn.Conv2d(192, 32, 1)
		self.conv5 = nn.Conv2d(320, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()
		
		# feature map 1
		self.f_map1_conv1 = nn.Conv2d(2048, 32, 1)
		self.f_map1_bn1 = nn.BatchNorm2d(32)
		self.f_map1_relu1 = nn.ReLU()
		self.f_map1_conv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.f_map1_bn2 = nn.BatchNorm2d(32)
		self.f_map1_relu2 = nn.ReLU()

		# feature map 2
		self.f_map2_conv1 = nn.Conv2d(128, 32, 1)
		self.f_map2_bn1 = nn.BatchNorm2d(32)
		self.f_map2_relu1 = nn.ReLU()
		self.f_map2_conv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.f_map2_bn2 = nn.BatchNorm2d(32)
		self.f_map2_relu2 = nn.ReLU()

		# feature map 3
		self.f_map3_conv1 = nn.Conv2d(64, 32, 1)
		self.f_map3_bn1 = nn.BatchNorm2d(32)
		self.f_map3_relu1 = nn.ReLU()
		self.f_map3_conv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.f_map3_bn2 = nn.BatchNorm2d(32)
		self.f_map3_relu2 = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		#x[3] = 2048, 16, 16

		f_map1 = F.interpolate(x[3], scale_factor=8, mode='bilinear', align_corners=True)
		f_map1 = self.f_map1_relu1(self.f_map1_bn1(self.f_map1_conv1(f_map1)))
		f_map1 = self.f_map1_relu2(self.f_map1_bn2(self.f_map1_conv2(f_map1)))

		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True) #y = 2048, 32, 32
		y = torch.cat((y, x[2]), 1) #y = 3072, 32, 32
		y = self.relu1(self.bn1(self.conv1(y)))	#y = 128, 32, 32	
		y = self.relu2(self.bn2(self.conv2(y))) #y = 128, 32, 32

		f_map2 = F.interpolate(y, scale_factor=4, mode='bilinear', align_corners=True)
		f_map2 = self.f_map2_relu1(self.f_map2_bn1(self.f_map2_conv1(f_map2)))
		f_map2 = self.f_map2_relu2(self.f_map2_bn2(self.f_map2_conv2(f_map2)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True) #y = 128, 64, 64
		y = torch.cat((y, x[1]), 1) #y = 640, 64, 64
		y = self.relu3(self.bn3(self.conv3(y)))	#y = 64, 64, 64	 
		y = self.relu4(self.bn4(self.conv4(y))) #y = 64, 64, 64
		
		f_map3 = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		f_map3 = self.f_map3_relu1(self.f_map3_bn1(self.f_map3_conv1(f_map3)))
		f_map3 = self.f_map3_relu2(self.f_map3_bn2(self.f_map3_conv2(f_map3)))

		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True) #y = 64, 128, 128
		y = torch.cat((y, x[0]), 1) #y = 320, 128, 128
		y = self.relu5(self.bn5(self.conv5(y)))	#y = 32, 128, 128	
		y = self.relu6(self.bn6(self.conv6(y))) #y = 32, 128, 128
		
		y = self.relu7(self.bn7(self.conv7(y)))

		f_map_concat = torch.cat((f_map1,f_map2,f_map3,y), 1)
		return f_map_concat

class output(nn.Module):
	def __init__(self, scope=512):
		super(output, self).__init__()
		self.conv1 = nn.Conv2d(128, 1, 1)
		self.sigmoid1 = nn.Sigmoid()
		self.conv2 = nn.Conv2d(128, 4, 1)
		self.sigmoid2 = nn.Sigmoid()
		self.conv3 = nn.Conv2d(128, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 1024
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		score = self.sigmoid1(self.conv1(x))
		loc   = self.sigmoid2(self.conv2(x)) * self.scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
		geo   = torch.cat((loc, angle), 1) 
		return score, geo
		
	
class EAST(nn.Module):
	def __init__(self, pretrained=True):
		super(EAST, self).__init__()
		self.extractor = extractor(pretrained)
		self.merge     = merge()
		self.output    = output()
	
	def forward(self, x):
		return self.output(self.merge(self.extractor(x)))
		

if __name__ == '__main__':
	m = EAST()
	x = torch.randn(1, 3, 256, 256)
	score, geo = m(x)
	print(score.shape)
	print(geo.shape)
