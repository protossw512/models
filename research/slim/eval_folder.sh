CUDA_VISIBLE_DEVICES=0 python eval_folder_new.py \
                    --model_name=inception_resnet_v2 \
                    --checkpoint_path=/scratch2/wangxiny/workspace/models/slim/train_0412_1 \
                    --dataset_dir=/scratch2/wangxiny/bio/create_tfrecords/Val_0412_CE/hemispherebleb \
                    --dataset_name=bio \
