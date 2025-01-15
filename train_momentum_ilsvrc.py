import torch
import torchvision
from torchvision import models
import numpy as np
from dataset.cub200 import CUB200
from dataset.ilsvrc import ILSVRC2012, my_collate
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import Network, get_model
import time
from utils import AverageMeter, IOUFunciton_ILSRVC, compute_bboxes_from_scoremaps, intersect, visualize_heatmap, save_bbox_as_json
import cv2
from losses import ContrastiveLoss_fg_bg
from loss import *
#from torchsummary import summary
from PIL import Image





flag=True
db_name='ilvrsc'

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
                if db_name=='cub':
                    pred_boxes = pred_boxes_t[j]
                    pred_boxes = torch.from_numpy(np.array([pred_boxes[k][0] for k in range(len(pred_boxes))])).float()
                    gt_boxes = bboxes[:, 1:].float()

                    # calculate
                    inter = intersect(pred_boxes, gt_boxes)
                    area_a = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
                    area_b = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                    union = area_a + area_b - inter
                    IOU = inter / union
                else:
                    pred_boxes = pred_boxes_t[j]
                    gt_boxes = [bboxes[k][:, 1:] for k in range(len(bboxes))]

                    # calculate
                    IOU = IOUFunciton_ILSRVC(pred_boxes, gt_boxes)
                IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
                IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))

                Corcorrect[j] += IOU.sum()

            if i%100==3:
                visualize_heatmap(1, 'contrastive_dino', input.clone().detach(), ccam, cls_name, img_name,
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
    torch.save(
                {"state_dict": model.state_dict(),
                 "epoch": epoch + 1,
                 }, '{}/checkpoints/current_ilsvrc_epoch_{}.pth'.format('debug', epoch + 1))

    if current_best_CorLoc<50:
        flag=False

    return current_best_CorLoc, current_best_CorLoc_threshold


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))



if db_name=='cub':
    db_root='/home/milad/Projects/class_segment_fire/dataset/CUB-200-2011/CUB_200_2011'
    num_classes=200
else:
    db_root = '/home/devel/dev/data2/datasets/ilsvrc'
    num_classes = 1000

BATCH_SIZE=55
EPOCHS = 3
WORKERS=4
model_backbone='dinov2'

if model_backbone=='resnet':
    img_train_size=256
    img_train_crop=224
    img_test_size=480
    img_test_crop=448
else:
    
    img_train_size=256
    img_train_crop=224
    img_test_size=480
    img_test_crop=448


criterion = [SimMaxLoss(metric='cos', alpha=0.05).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=0.05).cuda()]

train_transforms = transforms.Compose([
    transforms.Resize(size=(img_train_size, img_train_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(img_train_crop, img_train_crop)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_transforms_p = transforms.Compose([
    transforms.Resize(size=(img_train_size, img_train_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-30,30)),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomCrop(size=(img_train_crop, img_train_crop)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# we follow PSOL to adopt 448x448 as input to generate pseudo bounding boxes
test_transforms = transforms.Compose([
    transforms.Resize(size=(img_test_size, img_test_size)),
    transforms.CenterCrop(size=(img_test_crop, img_test_crop)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


if db_name=='cub':
    train_data = CUB200(root=db_root, input_size=256, crop_size=256, train=True, transform=train_transforms, p_transform=train_transforms_p)
    test_data = CUB200(root=db_root, input_size=480, crop_size=448, train=False, transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
else:
    train_data = ILSVRC2012(root=db_root, input_size=img_train_size, crop_size=img_train_crop, train=True, transform=train_transforms, p_transform=train_transforms_p)
    test_data = ILSVRC2012(root=db_root, input_size=img_test_size, crop_size=img_test_crop, train=False, transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True, collate_fn=my_collate)


num_test_images=len(test_loader)

print('num test images', num_test_images)





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

if model_backbone=='resnet':
    cin=2048+1024
else:
    cin=384

q_size=100000
m=.99
adaptive_q_size=0
max_q_size=26000
sgd_lr=.08

#Define model
#model=Network(pretrained='detco', cin=2048+1024)
model = get_model(pretrained='mocov2', cin=cin).cuda()
model_k = get_model(pretrained='mocov2', cin=cin).cuda()

model = model.to(device)


fg_q=torch.empty(size=(1,cin)).to(device)
bg_q=torch.empty(size=(1, cin)).to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=sgd_lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


num_iters = len(train_loader)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iters * config.EPOCHS)


loss_fn = ContrastiveLoss_fg_bg(.02, num_sim=10, num_dis_sim=30)
cross_entropy_loss = nn.CrossEntropyLoss()
loss_reg=.05


epoch_number = 0
best_vloss = 1_000_000.


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    global fg_q
    global bg_q
    global q_size
    if adaptive_q_size:
        q_size=q_size+7000
        if q_size>max_q_size:
            q_size=max_q_size
    

    #optimizer.zero_grad()
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        img, img_p, labels, _, _ = data
        img = img.to(device)
        img_p = img_p.to(device)
        labels = labels.to(device)
        sd = model.state_dict()
        sdk = model_k.state_dict()

        # Average all parameters
        for key in sd:
            sdk[key] = m * sdk[key] + (1 - m) * sd[key]
        model_k.load_state_dict(sdk)


        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        with torch.set_grad_enabled(True):
            fg_feats1, bg_feats1, ccam1, cls_logits = model(img)
            model.backbone.train(False)
            model_k.train(False)
            fg_feats2, bg_feats2, ccam2, cls_logits2 = model_k(img_p)

            fg_feats2=fg_feats2.detach()
            bg_feats2=bg_feats2.detach()
            fg_q = torch.cat([fg_q, fg_feats2], dim=0)[
                   -min(q_size, fg_q.shape[0] + fg_feats2.shape[0] - 1):]

            bg_q = torch.cat([bg_q, bg_feats2], dim=0)[
                   -min(q_size, bg_feats2.shape[0] + bg_q.shape[0] - 1):]
            #print('q_size', fg_q.shape[0])
            # Compute the loss and its gradients
            loss = loss_fn(fg_feats1, bg_feats1, fg_q, bg_q)+ loss_reg*cross_entropy_loss(cls_logits, labels)

            
            loss.backward()

            # Adjust learning weights
            #if (i+1) % 1 ==0:
            optimizer.step()
            

        # Gather data and report

        scheduler.step()
        running_loss += loss.item()
        if i % 100 == 0:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def extract(db_root, db_name, train_loader, model, threshold):

    # set up the averagemeters
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # record the time
    end = time.time()

    total = 0

    # testing
    with torch.no_grad():
        for i, (input, target, cls_name, img_name) in enumerate(train_loader):

            # data to gpu
            input = input.cuda()

            # inference the model
            fg_feats, bg_feats, ccam, cls_logits = model(input)
            if flag:
                ccam = 1 - ccam

            pred_boxes_t = []  # x0,y0, x1, y1
            for j in range(input.size(0)):
                estimated_boxes_at_each_thr, _ = compute_bboxes_from_scoremaps(
                    ccam[j, 0, :, :].detach().cpu().numpy().astype(np.float32), [threshold], input.size(-1) / ccam.size(-1),
                    multi_contour_eval=False)
                pred_boxes_t.append(estimated_boxes_at_each_thr[0])

            total += input.size(0)

            pred_boxes = pred_boxes_t
            experiment_name='momentum_wsol_ilsvrc'
            # save predicted bboxes
            #save_bbox_as_json(db_root, db_name, experiment_name, i, 0, pred_boxes, cls_name, img_name)

            # print the current testing status
            if i % 100 == 0:
                print('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    0, i, len(train_loader), batch_time=batch_time), flush=True)
                # image debug


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))
    model.train(True)
    model_k.train(False)
    #momentum update



    avg_loss = train_one_epoch(epoch, writer)

    model.eval()
    best_CorLoc, best_threshold =test(test_loader, model, epoch)
    global_best_threshold = best_threshold


train_transforms = transforms.Compose([
        transforms.Resize(size=(480, 480)),
        transforms.CenterCrop(size=(448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

train_data = ILSVRC2012(root=db_root, input_size=480, crop_size=448, train=True, transform=train_transforms)


train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=False)
extract(db_root, db_name, train_loader, model, global_best_threshold)
