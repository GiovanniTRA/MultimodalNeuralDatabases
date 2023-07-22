#! /bin/bash
# This script is used to generate the results in Table 1 of the paper.

TASKTYPE="retriever"
QUERYTYPE="count"
STOP_ALGO_MODE="mixed"
DEVICE="cuda:0"
OBJID=0
RESULT_DIR="../results/tab1"

#['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

MODEL="RN50"
STOP_ALGO="../support-materials/stop-algo/RN50_StopAlgo.pt"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.k=10 retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE

MODEL="RN101"
STOP_ALGO="../support-materials/stop-algo/RN101_StopAlgo.pt"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.k=10 retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE

MODEL="RN50x4"
STOP_ALGO="../support-materials/stop-algo/RN50x4_StopAlgo.pt"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.k=10 retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE

MODEL="RN50x16"
STOP_ALGO="../support-materials/stop-algo/RN50x16_StopAlgo.pt"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.k=10 retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE

MODEL="RN50x64"
STOP_ALGO="../support-materials/stop-algo/RN50x64_StopAlgo.pt"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.k=10 retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE

MODEL="ViT-B/32"
STOP_ALGO="../support-materials/stop-algo/ViT-B32_StopAlgo.pt"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.k=10 retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE

MODEL="ViT-L/14"
STOP_ALGO="../support-materials/stop-algo/ViT-L14_StopAlgo.pt"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.k=10 retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE

MODEL="ViT-L/14@336px"
STOP_ALGO="../support-materials/stop-algo/ViT-L14@336px_StopAlgo.pt"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.k=10 retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE