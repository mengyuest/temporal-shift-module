python imta_main_GE.py \
--data-root "[PATH_TO_IMAGENET]" \
--data dummy \
--gpu 0,2,3,6 \
--arch msdnet_ge \
--batch-size 256 \
--epochs 90 \
--nBlocks 5 \
--stepmode even \
--growthRate 16 \
--grFactor 1-2-4-4 \
--bnFactor 1-2-4-4 \
--step 4 \
--base 4 \
--nChannels 32 \
--use-valid \
-j 8
