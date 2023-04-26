DATASET_PATH=../data
SIFTSMALL_PATH=$DATASET_PATH/siftsmall
BIN_PATH=../build/examples

NNSTRATEGY=HEAP
# NNSTRATEGY=EA_TI100var0.9
VAQPARAM=VAQ256m32min7max8var1
METHOD=${VAQPARAM},${NNSTRATEGY}
REFINE=100,200

$BIN_PATH/demo_vaq --dataset $SIFTSMALL_PATH/siftsmall_base.fvecs\
  --queries $SIFTSMALL_PATH/siftsmall_query.fvecs\
  --file-format-ori fvecs\
  --timeseries-size 128\
  --dataset-size 10000\
  --queries-size 100\
  --result answer/answer_vaq_${METHOD}_refine${REFINE}_sift_10K.csv\
  --groundtruth $SIFTSMALL_PATH/siftsmall_groundtruth.ivecs\
  --groundtruth-format ivecs\
  --method $METHOD\
  --k 100\
  --refine $REFINE
