import torch
import torchvision
from torchvision import models
from dataset.cub200 import CUB200
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from model import Network, get_model
import time
from utils import AverageMeter, compute_bboxes_from_scoremaps, intersect, visualize_heatmap
import cv2
from losses import ContrastiveLoss_fg_bg
from loss import *
#from torchsummary import summary
from PIL import Image




flag=True


def test(test_loader, model, epoch):

    # set up the averagemeters
    NUM_THRESHOLD=20
    PRINT_FREQ=25
    batch_time = AverageMeter()
    losses = AverageMeter()
    threshold = [(i + 1) / NUM_THRESHOLD for i in range(NUM_THRESHOLD - 1)]
    print('current threshold list: {}'.format(threshold))

    # switch to evaluate mode
    model.eval()
    global flag
    # record the time
    end = time.time()

    total = 0
    Corcorrect = torch.Tensor([[0] for i in range(len(threshold))])

    # testing
    with torch.no_grad():
        for i, (input, target, bboxes, cls_name, img_name) in enumerate(test_loader):

            # data to gpu
            input = input.cuda()
            target = target.cuda()


            # inference the model
            fg_feats, bg_feats, ccam = model(input)

            if flag:
                ccam = 1 - ccam

            pred_boxes_t = [[] for j in range(len(threshold))]  # x0,y0, x1, y1
            for j in range(input.size(0)):

                estimated_boxes_at_each_thr, _ = compute_bboxes_from_scoremaps(
                    ccam[j, 0, :, :].detach().cpu().numpy().astype(np.float32), threshold, input.size(-1)/ccam.size(-1), multi_contour_eval=False)

                for k in range(len(threshold)):
                    pred_boxes_t[k].append(estimated_boxes_at_each_thr[k])



            # acc1 = accuracy(main_out.data, target)[0]


            # measure elapsed time
            torch.cuda.synchronize()

            total += input.size(0)
            for j in range(len(threshold)):
                pred_boxes = pred_boxes_t[j]
                pred_boxes = torch.from_numpy(np.array([pred_boxes[k][0] for k in range(len(pred_boxes))])).float()
                gt_boxes = bboxes[:, 1:].float()

                # calculate
                inter = intersect(pred_boxes, gt_boxes)
                area_a = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
                area_b = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                union = area_a + area_b - inter
                IOU = inter / union
                IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
                IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))

                Corcorrect[j] += IOU.sum()

            if i%20==1:
                if i % 20 == 1:
                    visualize_heatmap(1, 'contrastive', input.clone().detach(), ccam, cls_name, img_name,
                                      phase='test', bboxes=pred_boxes_t[NUM_THRESHOLD // 2], gt_bboxes=bboxes)

            batch_time.update(time.time() - end)
            end = time.time()

            # print the current testing status

    current_best_CorLoc = 0
    current_best_CorLoc_threshold = 0
    for i in range(len(threshold)):
        if (Corcorrect[i].item() / total) * 100 > current_best_CorLoc:
            current_best_CorLoc = (Corcorrect[i].item() / total) * 100
            current_best_CorLoc_threshold = threshold[i]

    print('Current => Correct: {:.2f}, threshold: {}'.format(current_best_CorLoc, current_best_CorLoc_threshold))

    return current_best_CorLoc, current_best_CorLoc_threshold


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))


db_root='/home/milad/Projects/class_segment_fire/dataset/CUB-200-2011/CUB_200_2011'
BATCH_SIZE=16
EPOCHS = 20
WORKERS=4
num_classes=200
model_backbone='resnet'


criterion = [SimMaxLoss(metric='cos', alpha=0.05).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=0.05).cuda()]

train_transforms = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_transforms_p = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-30,30)),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# we follow PSOL to adopt 448x448 as input to generate pseudo bounding boxes
test_transforms = transforms.Compose([
    transforms.Resize(size=(480, 480)),
    transforms.CenterCrop(size=(448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_data = CUB200(root=db_root, input_size=256, crop_size=256, train=True, transform=train_transforms, p_transform=train_transforms_p)
test_data = CUB200(root=db_root, input_size=480, crop_size=448, train=False, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

num_test_images=len(test_loader)

print('num test images', num_test_images)

train_loader_iter=iter(train_loader)
#img,label,_,_=next(train_loader_iter)
#print(img.shape)




''' classification models
if model_backbone=='mobilenet':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
elif model_backbone=='resnet':
    model=models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
'''

#Define model
#model=Network(pretrained='mocov2', cin=2048+1024)
model = get_model(pretrained='supervised').cuda()

print(model)
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


num_iters = len(train_loader)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters * config.EPOCHS)


loss_fn = ContrastiveLoss_fg_bg(.03)


epoch_number = 0
best_vloss = 1_000_000.


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.


    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        img, img_p, labels, _, _ = data
        img = img.to(device)
        img_p = img_p.to(device)
        labels = labels.to(device)


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        with torch.set_grad_enabled(True):
            fg_feats1, bg_feats1, ccam1 = model(img)
            fg_feats2, bg_feats2, ccam2 = model(img_p)
            fg_feats2=fg_feats2.detach()
            bg_feats2=bg_feats2.detach()

            # Compute the loss and its gradients
            loss = loss_fn(fg_feats1, bg_feats1, fg_feats2, bg_feats2)

            #loss1 = criterion[0](bg_feats1)
            #loss2 = criterion[1](bg_feats1, fg_feats1)
            #loss3 = criterion[2](fg_feats1)
            #loss = loss1 + loss2 + loss3
            loss.backward()

            # Adjust learning weights
            optimizer.step()

        scheduler.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 2 == 1:
            last_loss = running_loss / 2 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    model.train(True)
    #model.backbone.train(False)
    avg_loss = train_one_epoch(epoch, writer)

    model.eval()
    test(test_loader, model, epoch)


