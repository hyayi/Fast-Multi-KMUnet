from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from scipy.ndimage import zoom
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
import numpy as np
from typing import Tuple, Union
import torch
import torch.nn.functional as F


class Resize(BasicTransform):
    def __init__(self, new_size: Tuple[int, ...],**kwargs):
        super().__init__(**kwargs)
        self.new_size = new_size

    def _apply_to_image(self, image: torch.tensor, **kwargs) -> torch.tensor:
        return F.interpolate(image.unsqueeze(0),size = self.new_size,mode='bilinear',align_corners=False).squeeze(0)

    
    def _apply_to_segmentation(self, seg: torch.tensor, **kwargs) -> torch.tensor:
        return F.interpolate(seg.unsqueeze(0),size = self.new_size,mode='nearest').squeeze(0)

class Normalize(BasicTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _apply_to_image(self, image: torch.tensor, **kwargs) -> torch.tensor:
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8)
    
    def _apply_to_segmentation(self, seg: torch.tensor, **kwargs) -> torch.tensor:
        return seg

def get_training_transforms(
        image_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        mirror_axes: Tuple[int, ...],
) -> BasicTransform:
    transforms = []

    image_size_spatial = image_size
    ignore_axes = None
    transforms.append(Resize(new_size=image_size_spatial))

    transforms.append(
        SpatialTransform(
            image_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
            p_rotation=0.2,
            rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False  # , mode_seg='nearest'
        )
    )

    transforms.append(RandomTransform(
        GaussianNoiseTransform(
            noise_variance=(0, 0.1),
            p_per_channel=1,
            synchronize_channels=True
        ), apply_probability=0.1
    ))
    transforms.append(RandomTransform(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.),
            synchronize_channels=False,
            synchronize_axes=False,
            p_per_channel=0.5, benchmark=True
        ), apply_probability=0.2
    ))
    transforms.append(RandomTransform(
        MultiplicativeBrightnessTransform(
            multiplier_range=BGContrast((0.75, 1.25)),
            synchronize_channels=False,
            p_per_channel=1
        ), apply_probability=0.15
    ))
    transforms.append(RandomTransform(
        ContrastTransform(
            contrast_range=BGContrast((0.75, 1.25)),
            preserve_range=True,
            synchronize_channels=False,
            p_per_channel=1
        ), apply_probability=0.15
    ))
    transforms.append(RandomTransform(
        SimulateLowResolutionTransform(
            scale=(0.5, 1),
            synchronize_channels=False,
            synchronize_axes=True,
            ignore_axes=ignore_axes,
            allowed_channels=None,
            p_per_channel=0.5
        ), apply_probability=0.25
    ))
    transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=1,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ), apply_probability=0.1
    ))
    transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=0,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ), apply_probability=0.3
    ))
    if mirror_axes is not None and len(mirror_axes) > 0:
        transforms.append(
            MirrorTransform(
                allowed_axes=mirror_axes
            )
        )
    transforms.append(Normalize())

    return ComposeTransforms(transforms)

def get_validation_transforms(
        image_size: Union[np.ndarray, Tuple[int]],
) -> BasicTransform:
    transforms = []

    image_size_spatial = image_size
    transforms.append(Resize(new_size=image_size_spatial))
    transforms.append(Normalize())

    return ComposeTransforms(transforms)

if __name__ == "__main__":
    import nibabel as nib
    import os
    import pandas as pd
    import numpy as np 
    
    img_dir='/data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3006_active_learning_3004base/imagesTr'
    mask_dir='/data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3006_active_learning_3004base/labelsTr'
    img_ext='_0000.nii.gz'
    mask_ext='.nii.gz'
    cls_df_path='/data/image/project/ng_tube/nnunet/data/metafile/Dataset3006_label_version_3.00(25.08.14).csv'

    cls_df = pd.read_csv(cls_df_path)
    image_path = os.path.join(img_dir, cls_df.iloc[0]['image'] + img_ext)
    img = nib.load(image_path).get_fdata()[:,:].transpose(2,0,1)
    
    print("Original image shape:", img.shape)
    mask = nib.load(os.path.join(mask_dir, cls_df.iloc[0]['image'] + mask_ext)).get_fdata()[:,:].transpose(2,0,1)
    print("Original mask shape:", mask.shape)
    img = torch.tensor(img, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)
    print(img.shape)
    print(mask.shape)
    transform = get_training_transforms(image_size=(1024, 1024), rotation_for_DA=rotation_for_DA, mirror_axes=mirror_axes)
    data_dict = {'image': img, 'segmentation': mask}
    augmented = transform(**data_dict)
    img_aug = augmented['image']
    mask_aug = augmented['segmentation']
    print("Transformed image shape:", img_aug.shape)
    print("Transformed mask shape:", mask_aug.shape)