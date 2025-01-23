""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import numpy as np
import torch
from PIL import Image
import nibabel as nib
from torch.utils.data import Dataset
from pathlib import Path    
from func_3d.utils import random_click, generate_bbox
import json
import os


class MSD(Dataset):
    def __init__(self, args, dataset: str, fold: int, transform = None, transform_msk = None, mode = 'train', prompt = 'click', seed=None, variation=0):
        nnunet_raw = Path(os.environ.get("nnUNet_raw", "../data/nnUNet_raw")).resolve()
        nnunet_preprossed = Path(os.environ.get("nnUNet_preprocessed", "../data/nnUNet_preprocessed")).resolve()

        assert nnunet_raw.exists()
        assert nnunet_preprossed.exists()

        self.raw_folder = nnunet_raw / dataset 
        self.preprocessed_folder = nnunet_preprossed / dataset

        self.fold = fold

        with open(self.preprocessed_folder / "dataset.json", "r") as f:
            dataset_info = json.load(f)

        self.n_channels = len(dataset_info["channel_names"])

        with open(self.preprocessed_folder / "splits_final.json", "r") as f:
            splits = json.load(f)

        self.train_imgs = []
        self.val_imgs = []
        for name in splits[fold][mode]:
            for i in range(self.n_channels):
                # integer with 4 digits, i.e. leading zeros
                self.train_imgs += [f"{name}_{'%04d' % i}.nii.gz"]
            self.val_imgs += [f"{name}.nii.gz"]
        
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'train':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.train_imgs)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        img_path = self.raw_folder / "imagesTr" / self.train_imgs[index]
        mask_path = self.raw_folder / "labelsTr" / self.val_imgs[index]

        full_img = nib.load(img_path).get_fdata()    # type: ignore
        full_mask = nib.load(mask_path).get_fdata()    # type: ignore

        assert full_img.shape == full_mask.shape, f"Image and mask shape do not match"

        data_seg_3d = full_mask.copy()

        # We find the first frame that contains a foreground label
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:
                data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i

        # We find the last frame that contains a foreground label
        for j in reversed(range(data_seg_3d.shape[-1])):
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j+1]
                break

        num_frame = data_seg_3d.shape[-1]
        if self.video_length is None:
            video_length = int(num_frame / 4)
        else:
            video_length = self.video_length
        if num_frame > video_length and self.mode == 'train':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        for frame_index in range(starting_frame, starting_frame + video_length):
            img_grayscale = full_img[..., frame_index + starting_frame_nonzero]
           
            # We need to convert to RGB
            img = np.stack([img_grayscale, img_grayscale, img_grayscale], axis=-1)
            mask = data_seg_3d[..., frame_index]

            obj_list = np.unique(mask[mask > 0])
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)


            img = Image.fromarray(img.astype(np.uint8))  
            img = img.resize(newsize, resample=Image.BILINEAR)  
            img = np.array(img)  
            img = torch.tensor(np.array(img, dtype=np.uint8)).permute(2, 0, 1)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict


        image_meta_dict = {'filename_or_obj':self.train_imgs[index]}
        if self.prompt == 'bbox':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict':image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'p_label':point_label_dict,
                'pt':pt_dict,
                'image_meta_dict':image_meta_dict,
            }