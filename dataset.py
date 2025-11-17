import os

import cv2
import numpy as np
import torch
import torch.utils.data
import nibabel as nib
import os
import pandas as pd
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,img_dir, mask_dir, img_ext, mask_ext, target_size,cls_df_path,mode='train', transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.cls_df = self._load_df(cls_df_path,mode)
        self.class_mapping_table = {'complete':0, 'incomplete':1}
        self.target_size = target_size

    def _load_df(self,cls_df_path,mode):
        cls_df = pd.read_csv(cls_df_path)
        cls_df = cls_df[cls_df['type'] == mode]
        cls_df = cls_df.reset_index(drop=True)
        cls_df = cls_df[['image','class_new']]
        cls_df = cls_df[cls_df['class_new'].isin(['complete','incomplete'])].reset_index(drop=True)
        return cls_df
    def __len__(self):
        return len(self.cls_df)

    def __getitem__(self, idx):
        img_id = self.cls_df.iloc[idx]['image']
        cls_target = self.class_mapping_table[self.cls_df.iloc[idx]['class_new']]

        nii_img = nib.load(os.path.join(self.img_dir, img_id + self.img_ext))
        nii_mask = nib.load(os.path.join(self.mask_dir, img_id + self.mask_ext))

        orig_zooms = nii_img.header.get_zooms()
        orig_shape = nii_img.shape

        orig_sp_x, orig_sp_y = orig_zooms[0], orig_zooms[1]
        orig_dim_x, orig_dim_y = orig_shape[0], orig_shape[1]

        # 1024x1024로 변환 시 각 축의 스케일 비율 계산
        # (원본 길이 / 목표 길이 1024)
        scale_x = orig_dim_x / self.target_size[1]
        scale_y = orig_dim_y / self.target_size[0]

        # 변환된 Spacing 계산
        new_sp_x = orig_sp_x * scale_x
        new_sp_y = orig_sp_y * scale_y
        # 读取图像
        img = nii_img.get_fdata()[:,:].transpose(2,0,1)
        img = torch.tensor(img, dtype=torch.float32)
        # 读取掩码
        mask = nii_mask.get_fdata()[:,:].transpose(2,0,1)
        mask = torch.tensor(mask, dtype=torch.float32)

        # 如果使用了数据增强，则应用变换
        if self.transform is not None:
            augmented = self.transform(**{'image': img, 'segmentation': mask})
            img = augmented['image']
            mask = augmented['segmentation']

        # 显式指定数据类型
        spacing_tensor = torch.tensor([new_sp_x, new_sp_y], dtype=torch.float32)
        cls_target = torch.tensor(cls_target, dtype=torch.long)  # 分类目标张量
        return img, mask, cls_target,spacing_tensor, {'img_id': img_id}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        from get_transforms import get_training_transforms
    except ImportError:
        print("Could not import get_transforms.py. Make sure it exists.")
        exit()

    # --- 2. Setup (Modify paths as needed) ---
    IMG_DIR = '/data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3006_active_learning_3004base/imagesTr'
    MASK_DIR = '/data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3006_active_learning_3004base/labelsTr'
    CSV_PATH = '/data/image/project/ng_tube/nnunet/data/metafile/Dataset3006_label_version_3.00(25.08.14).csv'
    IMG_EXT = '_0000.nii.gz'
    MASK_EXT = '.nii.gz'
    MODE = 'val'
    OUTPUT_DIR = '/workspace/debug_img' # Output directory for debug images
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 3. Define 3D-mode transforms for batchgeneratorsv2 ---
    # (이전 RuntimeError를 피하기 위해 3D 모드 사용)
    rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
    mirror_axes = (0, 1) # 2D mirror (H, W) -> (D, H, W) 3D 공간에서는 (1, 2)
    
    # 2D 이미지 크기를 3D 공간 (D, H, W)으로 정의합니다.
    spatial_image_size = (1024, 1024) 
    
    transform = get_training_transforms(
        image_size=spatial_image_size, 
        rotation_for_DA=rotation_for_DA, 
        mirror_axes=(0, 1) # (D, H, W)에서 H와 W축을 미러링
    )

    # --- 4. Create Dataset ---
    dataset = Dataset(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        img_ext=IMG_EXT,
        mask_ext=MASK_EXT,
        num_classes=2,
        cls_df_path=CSV_PATH,
        mode=MODE,
        transform=transform # batchgeneratorsv2 transform 전달
    )

    # --- 5. Loop and Save Debug Images ---
    num_samples_to_check = 20
    print(f"\n--- Starting Debug Image Generation (Saving to {OUTPUT_DIR}) ---")
    
    # DataLoader를 사용하지 않고 Dataset을 직접 순회
    for i in range(min(num_samples_to_check, len(dataset))):
        # __getitem__을 직접 호출하여 (C,H,W) 텐서를 받음
        img_tensor, mask_tensor, cls_target, info = dataset[i]
        
        img_id = info['img_id']
        class_name = 'complete' if cls_target.item() == 0 else 'incomplete'

        # --- Convert Tensors to NumPy for plotting ---
        # (1, H, W) -> (H, W)
        image_slice = img_tensor.numpy().squeeze()
        mask_slice = mask_tensor.numpy().squeeze()
        
        # --- Save 1: Augmented Image Only ---
        save_path_img = os.path.join(OUTPUT_DIR, f"{img_id}__{class_name}__image.png")
        plt.figure(figsize=(10, 10))
        # batchgenerators Normalize는 Z-score (평균 0, 표준 1)이므로
        # vmin/vmax를 설정하여 대비를 명확하게 봅니다.
        plt.imshow(image_slice, cmap='gray', vmin=-3, vmax=3) 
        plt.title(f"Image Only\n{img_id} (Class: {class_name})")
        plt.axis('off')
        plt.savefig(save_path_img, bbox_inches='tight')
        plt.close()

        # --- Save 2: Image + Mask Overlay ---
        save_path_overlay = os.path.join(OUTPUT_DIR, f"{img_id}__{class_name}__overlay.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(image_slice, cmap='gray', vmin=-3, vmax=3)
        # 마스크를 반투명한 빨간색으로 오버레이
        plt.imshow(mask_slice, cmap='Reds', alpha=0.3, vmin=0, vmax=1)
        plt.title(f"Image + Mask Overlay\n{img_id} (Class: {class_name})")
        plt.axis('off')
        plt.savefig(save_path_overlay, bbox_inches='tight')
        plt.close()
        
        print(f"Saved debug images for {img_id} (Sample {i+1}/{num_samples_to_check})")


    print(f"\n--- Debug Image Generation Finished ---")