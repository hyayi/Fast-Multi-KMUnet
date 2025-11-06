# dataset=busi
# input_size=256
# python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
# python val.py --name ${dataset}_UKAN 

# dataset=glas
# input_size=512
# python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
# python val.py --name ${dataset}_UKAN 

# dataset=cvc
# input_size=256
# python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
# python val.py --name ${dataset}_UKAN 

dataset=ngtube
input_size=1024
python /workspace/my/FastKM_UNet_cls/train.py --arch UKANCls --dataset ${dataset} --input_w 256 --input_h 256 --name ${dataset}_UKAN_cls_test \
                --image_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3006_active_learning_3004base/imagesTr \
                --mask_dir /data/image/project/ng_tube/nnunet/data/nnUNet_raw/Dataset3006_active_learning_3004base/labelsTr \
                --splits_final /data/image/project/ng_tube/nnunet/data/nnUNet_preprocessed/Dataset3006_active_learning_3004base/splits_final.json \
                --cls_df_path "/data/image/project/ng_tube/nnunet/data/metafile/Dataset3006_label_version_3.00(25.08.14).csv" \
                -b 64
# python val.py --name ${dataset}_UKAN 






