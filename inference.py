from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from model import get_model
from utils import visualize_heatmap
import cv2
import cmapy
import os

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list, factor,
                                  multi_contour_eval=False):
    """
    Copy from: https://github.com/clovaai/wsolevaluation
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)

            estimated_boxes.append([x0 * factor, y0 * factor, x1 * factor, y1 * factor])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


def visualize_heatmap(image, attmaps, cls_name, image_name):
    _, c, h, w = image.shape
    image=image.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[..., ::-1]
    bbox_image = image.copy()
    attmap = attmaps.squeeze().to('cpu').detach().numpy()
    attmap = attmap / np.max(attmap)
    estimated_boxes, _ = compute_bboxes_from_scoremaps(attmap, [.5], 14)
    attmap = np.uint8(attmap * 255)
    # colormap = cv2.applyColorMap(cv2.resize(attmap, (w, h)), cv2.COLORMAP_JET)
    colormap = cv2.applyColorMap(cv2.resize(attmap, (w, h)), cmapy.cmap('seismic'))
    print('color map shapes are', colormap.shape)
    print('image shapes are', image.shape)
    cam = colormap + 0.5 * image
    cam = cam / np.max(cam)
    cam = np.uint8(cam * 255).copy()
    if not os.path.exists('debug/images/manuals/colormaps/{}'.format(cls_name)):
        os.mkdir('debug/images/manuals/colormaps/{}'.format(cls_name))
    cv2.imwrite(f'debug/images/manuals/colormaps/{cls_name}/{image_name}_heatmap.jpg', cam)

    show_bbox=1
    gt_bboxes=None
    if show_bbox:
        box = estimated_boxes[0][0]
        print('box is', box)
        cv2.rectangle(bbox_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)  # BGR
    cv2.imwrite(f'debug/images/manuals/colormaps/{cls_name}/{image_name}_bbox.jpg', bbox_image)







img_test_size=480
img_test_crop=448
cin=384
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(pretrained='mocov2', cin=cin).cuda()
model.load_state_dict(torch.load('debug/checkpoints/current_ilsvrc_epoch_2.pth')["state_dict"])
model = model.to(device)

model.eval()  # Set the model to evaluation mode

transform = transforms.Compose([
    transforms.Resize((img_test_crop, img_test_crop)),  # Resize image to match model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize image
])
cls_name="n04026417"
img_name="ILSVRC2012_val_00018579.JPEG"
image_path = '/home/devel/dev/data2/datasets/ilsvrc/val/'+cls_name+'/'+img_name
print("image path:", image_path)
image = Image.open(image_path)
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Pass the image through the model
with torch.no_grad():
    fg_feats, bg_feats, ccam = model(image)
    ccam=1-ccam
    visualize_heatmap(image, ccam, cls_name, img_name)
