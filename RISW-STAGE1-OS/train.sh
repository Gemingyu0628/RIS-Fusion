CUDA_VISIBLE_DEVICES=0 python train_stage1_gmy.py \
    --batch_size 1\
    --size 320 \
    --dataset IVF \
    --splitBy unc \
    --test_split val \
    --epoch 30 \
    --bert_tokenizer clip \
    --backbone clip-RN50 \
    --max_query_len 20 \
    --negative_samples 2 \
    --output ./weights/stage1/vis_ir_train \
    --pretrain resume/ckpt_320_epoch_13_best.pth\
    --resume\
    --board_folder ./output/board \

