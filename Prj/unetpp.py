import torch.utils.data as data
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from torch import nn
from torch.nn import functional as F
import torch
import torchvision
from metrics import *
from plot import loss_plot
from plot import metrics_plot

EPOCHS = 100
BATCH_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])
y_transforms = transforms.ToTensor()


class IsbiCellDataset:
    def __init__(self, state, transform=None, target_transform=None):
        self.state = state
        self.aug = True
        self.root = r'./isbi'
        self.img_paths = None
        self.mask_paths = None
        self.train_img_paths, self.val_img_paths, self.test_img_paths = None, None, None
        self.train_mask_paths, self.val_mask_paths, self.test_mask_paths = None, None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform

    def getDataPath(self):
        self.img_paths = glob(self.root + r'/train/images/*')
        self.mask_paths = glob(self.root + r'/train/label/*')
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = \
            train_test_split(self.img_paths, self.mask_paths,
                             test_size=0.2, random_state=41)
        self.test_img_paths, self.test_mask_paths = self.val_img_paths, self.val_mask_paths
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            return self.train_img_paths, self.train_mask_paths
        if self.state == 'val':
            return self.val_img_paths, self.val_mask_paths
        if self.state == 'test':
            return self.test_img_paths, self.test_mask_paths

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        pic = cv2.imread(pic_path)
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        pic = pic.astype('float32') / 255
        mask = mask.astype('float32') / 255
        if self.transform is not None:
            img_x = self.transform(pic)
        if self.target_transform is not None:
            img_y = self.target_transform(mask)
        return img_x, img_y, pic_path, mask_path

    def __len__(self):
        return len(self.pics)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class NestedUNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        self.sigmoid = nn.Sigmoid()

        self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output1 = self.final1(x0_1)
        output1 = self.sigmoid(output1)
        output2 = self.final2(x0_2)
        output2 = self.sigmoid(output2)
        output3 = self.final3(x0_3)
        output3 = self.sigmoid(output3)
        output4 = self.final4(x0_4)
        output4 = self.sigmoid(output4)

        return [output1, output2, output3, output4]


def getLog():
    filename = './log.log'
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return logging


def getModel():
    model = NestedUNet(3, 1).to(device)
    return model


def getDataset():
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    train_dataset = IsbiCellDataset(
        r'train', transform=x_transforms, target_transform=y_transforms)
    train_dataloaders = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataset = IsbiCellDataset(
        r"val", transform=x_transforms, target_transform=y_transforms)
    val_dataloaders = DataLoader(val_dataset, batch_size=1)
    test_dataset = IsbiCellDataset(
        r"test", transform=x_transforms, target_transform=y_transforms)
    test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders, val_dataloaders, test_dataloaders


def val(model, best_iou, val_dataloaders):
    model = model.eval()
    with torch.no_grad():
        i = 0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)
        for x, _, pic, mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            img_y = torch.squeeze(y[-1]).cpu().numpy()

            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0], img_y)
            dice_total += get_dice(mask[0], img_y)
            if i < num:
                i += 1
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_hd=%f,aver_dice=%f' %
              (aver_iou, aver_hd, aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' %
                     (aver_iou, aver_hd, aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(
                aver_iou, best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/UNETpp.pth')
        return best_iou, aver_iou, aver_dice, aver_hd


def train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs, threshold):
    best_iou, aver_iou, aver_dice, aver_hd = 0, 0, 0, 0
    num_epochs = epochs
    threshold = threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y, _, mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = 0
            for output in outputs:
                loss += criterion(output, labels)
            loss /= len(outputs)
            if threshold != None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) //
                                              train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) //
                                                     train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        best_iou, aver_iou, aver_dice, aver_hd = val(
            model, best_iou, val_dataloader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(num_epochs, loss_list)
    metrics_plot(num_epochs, 'iou&dice', iou_list, dice_list)
    metrics_plot(num_epochs, 'hd', hd_list)
    return model


def test(val_dataloaders, save_predict=False):
    logging.info('final test........')
    if save_predict == True:
        dir = r'./saved_predict'
        if not os.path.exists(dir):
            os.makedirs(dir)
    model.load_state_dict(torch.load(
        r'./saved_model/UNetpp.pth', map_location='cpu'))
    model.eval()

    with torch.no_grad():
        i = 0
        miou_total = 0
        hd_total = 0
        dice_total = 0
        num = len(val_dataloaders)
        for pic, _, pic_path, mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            predict = torch.squeeze(predict[-1]).cpu().numpy()

            iou = get_iou(mask_path[0], predict)
            miou_total += iou
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0], predict)
            dice_total += dice

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict, cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            if save_predict == True:
                plt.savefig(dir + '/' + mask_path[0].split('/')[-1])
            print('iou={},dice={}'.format(iou, dice))
            if i < num:
                i += 1
        print('Miou=%f,aver_hd=%f,dv=%f' %
              (miou_total/num, hd_total/num, dice_total/num))
        logging.info('Miou=%f,aver_hd=%f,dv=%f' %
                     (miou_total/num, hd_total/num, dice_total/num))


if __name__ == '__main__':
    logging = getLog()
    print('**************************')
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\n========' %
                 ("UNet++", str(EPOCHS), "4"))
    print('**************************')
    model = getModel()
    train_dataloaders, val_dataloaders, test_dataloaders = getDataset()
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    train(model, criterion, optimizer,
          train_dataloaders, val_dataloaders, EPOCHS, None)
    # test(test_dataloaders, save_predict=True)