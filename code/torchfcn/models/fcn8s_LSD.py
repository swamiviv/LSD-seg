import os.path as osp
import fcn
import torch
import torch.nn as nn
from torch.autograd import Variable
from model_utils import get_upsampling_weight
from model_utils import FCN8s   
from model_utils import ResnetBlock
 
class FCN8s_LSD(FCN8s):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8
        
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        fc7_response = self.drop7(h)

        h = self.score_fr(fc7_response)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h, fc7_response, pool4, pool3

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.nz = 0
        self.ndim1 = 4096
        self.ndim2 = 512
        self.ndim3 = 256
        ngf = 64
        self.stage1_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.ndim1 + self.nz, ngf*4, 7, 1, 0,bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
            ResnetBlock(ngf*4, norm_type=2, bias=False),
            ResnetBlock(ngf*4, norm_type=2, bias=False),
            nn.ConvTranspose2d(ngf*4, ngf * 8, 3, 2, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(True),
            ResnetBlock(ngf*8, norm_type=2, bias=False),
            ResnetBlock(ngf*8, norm_type=2, bias=False),
        )

        self.stage2_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.ndim2 + self.nz + ngf*8, ngf * 4, 3, 2, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
            ResnetBlock(ngf*4, norm_type=2, bias=False),
            ResnetBlock(ngf*4, norm_type=2, bias=False),
        )

        self.stage3_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.ndim3 + self.nz + ngf*4, ngf*2, 5, 2, 2, 0, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(True),
            ResnetBlock(ngf*2, norm_type=2, bias=False),
            ResnetBlock(ngf*2, norm_type=2, bias=False),
            nn.ConvTranspose2d(ngf*2, ngf, 5, 2, 2, 0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, norm_type=2, bias=False),
            ResnetBlock(ngf, norm_type=2, bias=False),
        )

        self.stage4_upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 8, 2, 100),
            nn.Sigmoid()
        )

    def forward(self, fc_response, pool4_response, pool3_response):   
        
        input_stage1 = fc_response
        if self.nz > 0:
            noise = torch.FloatTensor(input_stage1.size()[0],self.nz,input_stage1.size()[2],input_stage1.size()[3]).normal_(0,1)    
            noisev = Variable(noise.cuda()) 
            input_new = torch.cat((input_stage1, noisev),1)
        else:
            input_new = input_stage1
        output_stage1 = self.stage1_upsample(input_new)

        input_stage2 = torch.cat((output_stage1, pool4_response), 1)
        if self.nz > 0:
            noise = torch.FloatTensor(input_stage2.size()[0],self.nz,input_stage2.size()[2],input_stage2.size()[3]).normal_(0,1)    
            noisev = Variable(noise.cuda()) 
            input_new = torch.cat((input_stage2, noisev),1)
        else:
            input_new = input_stage2
        output_stage2 = self.stage2_upsample(input_new)

        input_stage3 = torch.cat((output_stage2, pool3_response), 1)
        if self.nz > 0:
            noise = torch.FloatTensor(input_stage3.size()[0],self.nz,input_stage3.size()[2],input_stage3.size()[3]).normal_(0,1)    
            noisev = Variable(noise.cuda()) 
            input_new = torch.cat((input_stage3, noisev),1)
        else:
            input_new = input_stage3
        
        output = self.stage3_upsample(input_new)        

        output = self.stage4_upsample(output)
        
        return output


"""
#resize convolution instead of transpose
class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.nz = 0
        self.ndim1 = 4096
        self.ndim2 = 512
        self.ndim3 = 256
        ngf = 64

        self.stage1_upsample = nn.Sequential(
            nn.Conv2d(self.ndim1 + self.nz, ngf*4, 3, 1, 4,bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
            ResnetBlock(ngf*4, norm_type=2, bias=False),
            ResnetBlock(ngf*4, norm_type=2, bias=False),
            torch.nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(ngf*4, ngf * 8, 4, 1, 1,bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(True),
            ResnetBlock(ngf*8, norm_type=2, bias=False),
            ResnetBlock(ngf*8, norm_type=2, bias=False),
        )

        self.stage2_upsample = nn.Sequential(
            torch.nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(self.ndim2 + self.nz + ngf*8, ngf * 2, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            ResnetBlock(ngf*2, norm_type=2, bias=False),
            ResnetBlock(ngf*2, norm_type=2, bias=False),
        )

        self.stage3_upsample = nn.Sequential(
            torch.nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(self.ndim3 + self.nz + ngf*2, ngf, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, norm_type=2, bias=False),
            torch.nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(ngf, ngf, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            ResnetBlock(ngf, norm_type=2, bias=False),
        )

        self.stage4_upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 8, 2, 100),
            nn.Sigmoid()
        )

    def forward(self, fc_response, pool4_response, pool3_response):

        input_stage1 = fc_response
        if self.nz > 0:
            noise = torch.FloatTensor(input_stage1.size()[0],self.nz,input_stage1.size()[2],input_stage1.size()[3]).normal_(0,1)
            noisev = Variable(noise.cuda())
            input_new = torch.cat((input_stage1, noisev),1)
        else:
            input_new = input_stage1
        output_stage1 = self.stage1_upsample(input_new)

        input_stage2 = torch.cat((output_stage1, pool4_response), 1)
        if self.nz > 0:
            noise = torch.FloatTensor(input_stage2.size()[0],self.nz,input_stage2.size()[2],input_stage2.size()[3]).normal_(0,1)
            noisev = Variable(noise.cuda())
            input_new = torch.cat((input_stage2, noisev),1)
        else:
            input_new = input_stage2
        output_stage2 = self.stage2_upsample(input_new)

        input_stage3 = torch.cat((output_stage2, pool3_response), 1)
        if self.nz > 0:
            noise = torch.FloatTensor(input_stage3.size()[0],self.nz,input_stage3.size()[2],input_stage3.size()[3]).normal_(0,1)
            noisev = Variable(noise.cuda())
            input_new = torch.cat((input_stage3, noisev),1)
        else:
            input_new = input_stage3
        output = self.stage3_upsample(input_new)
        output = self.stage4_upsample(output)
        return output
"""

class _netD(nn.Module):
    def __init__(self, n_class=19):
        nc = 3
        ndf = 64
        n_layers=3
        super(_netD, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(nc, 2*ndf, 4, 2, 2),    
            nn.InstanceNorm2d(2*ndf),
            nn.LeakyReLU(0.2, inplace=True),
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),                            
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),                            
            nn.MaxPool2d(2, 2),
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),                            
            nn.MaxPool2d(2, 2),
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),
            ResnetBlock(ndf*2, norm_type=2, bias=True, relu_type=2),                            
        )
        self.out_s = nn.Sequential(nn.Conv2d(ndf*2, 4, 3, padding=1))
        self.out_c = nn.Sequential(nn.Conv2d(ndf*2, n_class, 3, padding=1))

    def forward(self, input):   
        output = self.feature(input) 
        out_s = self.out_s(output)
        out_c = self.out_c(output)
        return out_s,out_c       

