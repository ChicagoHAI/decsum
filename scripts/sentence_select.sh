#!/bin/bash
set -x

OUTPUT_DIR=data/test_decsum
MODEL_PATH=${OUTPUT_DIR}/version_16-02-2022--18-59-02/checkpoints/epoch=2-val_loss=0.40.ckpt
RES_DIR=${OUTPUT_DIR}/models/sentence_select/

# DecSum
for m in Transformer # 
do
    for s in  window_1_DecSum_WD_sentbert #
    do
        for d in yelp # 
        do
            for k in 50trunc #10 # 5
            do
                # add hyperparameter search here
                for alpha in  1  
                do
                    for beta in 1  
                    do
                        for gamma in 1 
                        do 
                            (python -m models.sentence_select.main \
                                    --device 0 --feature_used notes \
                                    --trained_model_path ${MODEL_PATH} \
                                    --result_dir ${RES_DIR} \
                                    --data_dir ${OUTPUT_DIR} \
                                    --num_sentences $k \
                                    --alpha $alpha \
                                    --beta $beta \
                                    --gamma $gamma \
                                    --segment $s\_$k\_$alpha\_$beta\_$gamma \
                                    --model $m \
                                    --data $d \
                                    --num_review 50 \
                                    --target_type reg )
                        done
                    done
                done
            done
        done
    done
done
