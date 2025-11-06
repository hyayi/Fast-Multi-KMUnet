KM-UNet: KAN Mamba UNet for medical image segmentation
paper:https://arxiv.org/abs/2501.02559
Medical image segmentation is a critical task in medical imaging analysis. Traditional CNN-based methods struggle with 
modeling long-range dependencies, while Transformer-based models, despite their success, suffer from quadratic 
computational complexity. To address these limitations, we propose KM-UNet, a novel U-shaped network architecture that 
combines the strengths of Kolmogorov-Arnold Networks (KANs) and state-space models (SSMs). KM-UNet leverages the 
Kolmogorov-Arnold representation theorem for efficient feature representation and SSMs for scalable long-range modeling, 
achieving a balance between accuracy and computational efficiency. 
We evaluate KM-UNet on five benchmark datasets: ISIC17, ISIC18, CVC, BUSI, and GLAS. Experimental results 
demonstrate that KM-UNet achieves competitive performance compared to state-of-the-art methods in medical image 
segmentation tasks. 
To the best of our knowledge, KM-UNet is the first medical image segmentation framework integrating KANs and SSMs. 
This work provides a valuable baseline and new insights for the development of more efficient and interpretable medical 
image segmentation systems. 
Keywords:KAN,Manba, state-space models,UNet, Medical image segmentation, Deep learning

train:python train.py --arch KM_UNet --dataset {dataset} --input_w {input_size} --input_h {input_size} --name {dataset}_KM-UNet  --data_dir [YOUR_DATA_DIR]

For example:python train.py --arch KM_UNet --dataset busi --input_w 256 --input_h 256 --name busi_KM-UNet  --data_dir ./inputs

The input folder contains the five datasets used in the experiment,it is opening datasets,can download in Kaggle. 
The data format is:
Dataset name: 
	images     -1.png
	            -2.png
	            ...
	masks      -1.png
	           -2.png
	           ...
and the outputs folder contains the training data obtained from the article training(the weight file was larger than 25M,if you need,send me email)