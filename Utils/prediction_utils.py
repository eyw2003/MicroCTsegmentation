import cv2
import torch
from Utils.dataset_utils import normalize
import numpy as np
from tqdm import tqdm
import nibabel as nib
DEVICE ='cuda'
def convert_labels(label_volume):
    new_labels=np.zeros(label_volume.shape)
    for lbl in np.unique(label_volume):
        if lbl in LABELS_TO_KEEP:
            new_labels[np.where(label_volume==lbl)]=1
    return new_labels

def to_tensor(x, **kwargs):
    x= torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))
    return x
def predict_img(model, img):
    img = to_tensor(img)
    img = img.to(device=DEVICE)
    img = torch.unsqueeze(img, dim=0)
    pred_mask = torch.squeeze(model(img))
    pred_mask = pred_mask.detach().cpu().numpy()
    return pred_mask

def prepare_slice(img):
    img = normalize(img)
    img = img * 255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def predict_volume(model, volume, all_axials=False, preprocessing_fn=None):
    pred_vol = np.zeros(volume.shape)
    out_Z = np.zeros(volume.shape)
    if all_axials:
        out_X = np.zeros(volume.shape)
        out_Y = np.zeros(volume.shape)
    X, Y, Z = volume.shape
    for i in tqdm(range(Z)):
        img = prepare_slice(volume[:, :, i])
        pred_mask = predict_img(model, preprocessing_fn(img))
        out_Z[:, :, i] = pred_mask

    if all_axials:
        for i in tqdm(range(Y)):
            img = prepare_slice(volume[:, i, :])
            pred_mask = predict_img(model, preprocessing_fn(img))
            out_Y[:, i, :] = pred_mask
        for i in tqdm(range(X)):
            img = prepare_slice(volume[i, :, :])
            pred_mask = predict_img(model, preprocessing_fn(img))
            out_X[i, :, :] = pred_mask
        pred_vol = (out_X + out_Y + out_Z) / 3.0
        pred_vol[pred_vol > 0.5] = 1
        pred_vol[pred_vol < 0.5] = 0
        return pred_vol
    else:
        return out_Z
def save_mask_nii(volume_arr,affine,path):
    ni_img = nib.Nifti1Image(volume_arr,affine)
    nib.save(ni_img, path)