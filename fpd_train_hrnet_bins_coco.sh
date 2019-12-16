python tools/fpd_train.py \
    --tcfg experiments/fpd_coco/hrnet/w48_256x192_adam_lr1e-3.yaml `# experiments/fpd/hrnet/w48_256x192_adam_lr1e-3.yaml` \
    --cfg experiments/fpd_coco/hrnet/bin_256x192_adam_lr1e-3.yaml \
    GPUS '(0,1)' \
    DATASET.CACHE_ROOT 'data/cache' `# cache point path` \
    MODEL.INIT_WEIGHTS True \
    MODEL.IMAGE_SIZE 192,256 `# 288,384` \
    MODEL.HEATMAP_SIZE 48,64 `# 72,96` \
    MODEL.SIGMA 2 `# 3` \
    MODEL.EXTRA.STAGE2.NUM_CHANNELS 32,64 `# 48,96` \
    MODEL.EXTRA.STAGE3.NUM_CHANNELS 32,64,128 `# 48,96,192` \
    MODEL.EXTRA.STAGE4.NUM_CHANNELS 32,64,128,256 `# 48,96,192,384` \
    TRAIN.BATCH_SIZE_PER_GPU 18 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 60 `# 210` \
    TRAIN.LR 0.001 \
    TRAIN.LR_STEP 20,50 `# 170,200` \
    TRAIN.CHECKPOINT 'models/pytorch/pose_coco/pose_bin_256x192.pth' \
    TEST.BATCH_SIZE_PER_GPU 32 \
    TEST.USE_GT_BBOX False \
    DEBUG.DEBUG False \
    KD.TRAIN_TYPE 'FPD' \
    KD.TEACHER 'models/pytorch/pose_coco/pose_hrnet_w48_256x192.pth' \
    KD.ALPHA 0.5 \
