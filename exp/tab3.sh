#! /bin/bash

TASKTYPE="retriever"
QUERYTYPE="count"
DEVICE="cuda:0"
OBJID=0
MODEL="ViT-L/14@336px"
STOP_ALGO="../support-materials/stop-algo/ViT-L14@336px_StopAlgo.pt"


STOP_ALGO_MODE="threshold"
RESULT_DIR="../results/tab3_threshold"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.t=0.20

STOP_ALGO_MODE="topk"
RESULT_DIR="../results/tab3_topk"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10

STOP_ALGO_MODE="mixed"
RESULT_DIR="../results/tab3_mixed"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10

STOP_ALGO_MODE="stop_algo"
RESULT_DIR="../results/tab3_stop_algo"
python ../main.py data.obj_id=$OBJID task.task_type=$TASKTYPE task.query_type=$QUERYTYPE retriever.clip_model=$MODEL retriever.stop_algo_path=$STOP_ALGO experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10
