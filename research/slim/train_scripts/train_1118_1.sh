CUDA_VISIBLE_DEVICES=1 python ../train_image_classifier.py \
       --train_dir=../train_1118_1 \
       --dataset_name=bio \
       --dataset_split_name=train \
       --dataset_dir=/scratch2/wangxiny/bio/create_tfrecords/Padded_Images_Split_CE \
       --model_name=inception_resnet_v2 \
       --checkpoint_path=../inception_resnet_v2_2016_08_30.ckpt \
       --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
       --max_number_of_steps=15000 \
       --batch_size=32 \
       --learning_rate=0.0003 \
       --learning_rate_decay_type=exponential \
       --learning_rate_decay_factor=0.25 \
       --num_epochs_per_decay=100 \
       --save_interval_secs=1800 \
       --save_summaries_secs=600 \
       --log_every_n_steps=1 \
       --optimizer=rmsprop \
       --weight_decay=0.00004

#--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
