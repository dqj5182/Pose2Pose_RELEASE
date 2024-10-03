import torch
import torch.nn as nn
from torch.nn import functional as F
from torchgeometry import rotation_matrix_to_angle_axis
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls

import math
import copy
import smplx
import numpy as np



# Configuration
output_hm_shape = (8, 8, 8)
focal = (5000, 5000)
princpt = (128.0, 128.0)
camera_3d_size = 0.4
input_img_shape = (256, 256)
trainset_3d = ['Human36M']
trainset_2d = ['MSCOCO', 'MPII']



# MANO
class MANO(object):
    def __init__(self):
        self.layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
        self.layer = {'right': smplx.create('../data/base_data/human_models', 'mano', is_rhand=True, use_pca=False, flat_hand_mean=False, **self.layer_arg), 'left': smplx.create('../data/base_data/human_models', 'mano', is_rhand=False, use_pca=False, flat_hand_mean=False, **self.layer_arg)}
        self.vertex_num = 778
        self.face = {'right': self.layer['right'].faces, 'left': self.layer['left'].faces}
        self.shape_param_dim = 10

        if torch.sum(torch.abs(self.layer['left'].shapedirs[:,0,:] - self.layer['right'].shapedirs[:,0,:])) < 1:
            print('Fix shapedirs bug of MANO')
            self.layer['left'].shapedirs[:,0,:] *= -1

        # original MANO joint set
        self.orig_joint_num = 16
        self.orig_joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3')
        self.orig_root_joint_idx = self.orig_joints_name.index('Wrist')
        self.orig_flip_pairs = ()
        self.orig_joint_regressor = self.layer['right'].J_regressor.numpy() # same for the right and left hands

        # changed MANO joint set
        self.joint_num = 21 # manually added fingertips
        self.joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
        self.skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
        self.root_joint_idx = self.joints_name.index('Wrist')
        self.flip_pairs = ()
        # add fingertips to joint_regressor
        self.joint_regressor = self.transform_joint_to_other_db(self.orig_joint_regressor, self.orig_joints_name, self.joints_name)
        self.joint_regressor[self.joints_name.index('Thumb_4')] = np.array([1 if i == 745 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Index_4')] = np.array([1 if i == 317 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Middle_4')] = np.array([1 if i == 445 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Ring_4')] = np.array([1 if i == 556 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor[self.joints_name.index('Pinky_4')] = np.array([1 if i == 673 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)

    def transform_joint_to_other_db(self, src_joint, src_name, dst_name):
        src_joint_num = len(src_name)
        dst_joint_num = len(dst_name)

        new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
        for src_idx in range(len(src_name)):
            name = src_name[src_idx]
            if name in dst_name:
                dst_idx = dst_name.index(name)
                new_joint[dst_idx] = src_joint[src_idx]

        return new_joint


mano = MANO()




# ResNet
class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        
        self.load_state_dict(org_resnet)
        print("Initialize resnet from model zoo")







def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


# PositionNet
class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.joint_num = 21
        self.conv = make_conv_layers([2048,self.joint_num*output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_3d(self, heatmap3d):
        batch_size = heatmap3d.shape[0]
        depth, height, width = heatmap3d.shape[2:]
        heatmap3d = heatmap3d.reshape((batch_size, -1, depth*height*width))
        heatmap3d = F.softmax(heatmap3d, 2)
        heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

        accu_x = heatmap3d.sum(dim=(2,3))
        accu_y = heatmap3d.sum(dim=(2,4))
        accu_z = heatmap3d.sum(dim=(3,4))

        accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
        accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]
        accu_z = accu_z * torch.arange(depth).float().cuda()[None,None,:]

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

        coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
        return coord_out

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1,self.joint_num,output_hm_shape[0],output_hm_shape[1],output_hm_shape[2])
        joint_coord = self.soft_argmax_3d(joint_hm)
        return joint_coord


# RotationNet
class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.joint_num = 21
        self.mano_orig_joint_num = 16
        self.mano_shape_param_dim = 10
       
        # output layers
        self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)
        self.root_pose_out = self.make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
        self.pose_out = self.make_linear_layers([self.joint_num*(512+3), (self.mano_orig_joint_num-1)*6], relu_final=False) # without root joint
        self.shape_out = self.make_linear_layers([2048,self.mano_shape_param_dim], relu_final=False)
        self.cam_out = self.make_linear_layers([2048,3], relu_final=False)

    def make_linear_layers(self, feat_dims, relu_final=True, use_bn=False):
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

            # Do not use ReLU for final estimation
            if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
                if use_bn:
                    layers.append(nn.BatchNorm1d(feat_dims[i+1]))
                layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def sample_joint_features(self, img_feat, joint_xy):
        height, width = img_feat.shape[2:]
        x = joint_xy[:,:,0] / (width-1) * 2 - 1
        y = joint_xy[:,:,1] / (height-1) * 2 - 1
        grid = torch.stack((x,y),2)[:,:,None,:]
        img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:,:,:,0] # batch_size, channel_dim, joint_num
        img_feat = img_feat.permute(0,2,1).contiguous() # batch_size, joint_num, channel_dim
        return img_feat

    def forward(self, img_feat, joint_coord_img):
        batch_size = img_feat.shape[0]

        # shape parameter
        shape_param = self.shape_out(img_feat.mean((2,3)))

        # camera parameter
        cam_param = self.cam_out(img_feat.mean((2,3)))
        
        # pose parameter
        img_feat = self.conv(img_feat)
        img_feat_joints = self.sample_joint_features(img_feat, joint_coord_img)
        feat = torch.cat((img_feat_joints, joint_coord_img),2)
        root_pose = self.root_pose_out(feat.view(batch_size,-1))
        pose_param = self.pose_out(feat.view(batch_size,-1))
        
        return root_pose, pose_param, shape_param, cam_param






class Pose2Pose(nn.Module):
    def __init__(self, mode='test'):
        super(Pose2Pose, self).__init__()
        self.mode = mode

        self.mano_root_joint_idx = 0
        self.mano_orig_joint_num = 16
        self.mano_shape_param_dim = 10

        # hand networks
        self.backbone = ResNetBackbone(50) # 50 means ResNet-50
        self.position_net = PositionNet()
        self.rotation_net = RotationNet()

        # Initialization
        if self.mode == 'train':
            self.backbone.init_weights()
            self.position_net.apply(self.init_weights)
            self.rotation_net.apply(self.init_weights)

        self.mano_layer = copy.deepcopy(mano.layer['right']).cuda()
        self.trainable_modules = [self.backbone, self.position_net, self.rotation_net]

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight,std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight,std=0.01)
            nn.init.constant_(m.bias,0)
        
    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(focal[0]*focal[1]*camera_3d_size*camera_3d_size/(input_img_shape[0]*input_img_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def forward_position_net(self, inputs, backbone, position_net):
        img_feat = backbone(inputs['img'])
        joint_img = position_net(img_feat)
        return img_feat, joint_img
    
    def rot6d_to_axis_angle(self, x):
        batch_size = x.shape[0]

        x = x.view(-1,3,2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix
        
        rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).cuda().float()],2) # 3x4 rotation matrix
        axis_angle = rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
        axis_angle[torch.isnan(axis_angle)] = 0.0
        return axis_angle
    
    def forward_rotation_net(self, img_feat, joint_img, rotation_net):
        batch_size = img_feat.shape[0]

        # parameter estimation
        root_pose_6d, pose_param_6d, shape_param, cam_param = rotation_net(img_feat, joint_img)
        # change 6d pose -> axis angles
        root_pose = self.rot6d_to_axis_angle(root_pose_6d).reshape(-1,3)
        pose_param = self.rot6d_to_axis_angle(pose_param_6d.view(-1,6)).reshape(-1,(self.mano_orig_joint_num-1)*3)
        cam_trans = self.get_camera_trans(cam_param)
        return root_pose, pose_param, shape_param, cam_trans

    def get_coord(self, params, mode):
        batch_size = params['root_pose'].shape[0]

        output = self.mano_layer(global_orient=params['root_pose'], hand_pose=params['hand_pose'], betas=params['shape'])
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        joint_cam = torch.bmm(torch.from_numpy(mano.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
        root_joint_idx = self.mano_root_joint_idx

        # project 3D coordinates to 2D space
        cam_trans = params['cam_trans']
        if mode == 'train' and len(trainset_3d) == 1 and trainset_3d[0] == 'AGORA' and len(trainset_2d) == 0: # prevent gradients from backpropagating to SMPL/MANO/FLAME paraemter regression module
            x = (joint_cam[:,:,0].detach() + cam_trans[:,None,0]) / (joint_cam[:,:,2].detach() + cam_trans[:,None,2] + 1e-4) * focal[0] + princpt[0]
            y = (joint_cam[:,:,1].detach() + cam_trans[:,None,1]) / (joint_cam[:,:,2].detach() + cam_trans[:,None,2] + 1e-4) * focal[1] + princpt[1]
        else:
            x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * focal[0] + princpt[0]
            y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * focal[1] + princpt[1]
        x = x / input_img_shape[1] * output_hm_shape[2]
        y = y / input_img_shape[0] * output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam

        # add camera translation for the rendering
        mesh_cam = mesh_cam + cam_trans[:,None,:]
        return joint_proj, joint_cam, mesh_cam

    def forward(self, inputs):
        # network forward and get outputs
        # hand network
        img_feat, joint_img = self.forward_position_net(inputs, self.backbone, self.position_net)
        mano_root_pose, mano_hand_pose, mano_shape, cam_trans = self.forward_rotation_net(img_feat, joint_img.detach(), self.rotation_net)
        joint_proj, joint_cam, mesh_cam = self.get_coord({'root_pose': mano_root_pose, 'hand_pose': mano_hand_pose, 'shape': mano_shape, 'cam_trans': cam_trans}, self.mode)
        mano_hand_pose = mano_hand_pose.view(-1,(self.mano_orig_joint_num-1)*3)
        mano_pose = torch.cat((mano_root_pose, mano_hand_pose),1)

        # test output
        out = {'cam_trans': cam_trans} 

        out['img'] = inputs['img']
        out['joint_img'] = joint_img 
        out['mano_mesh_cam'] = mesh_cam
        out['mano_pose'] = mano_pose
        out['mano_shape'] = mano_shape

        return out