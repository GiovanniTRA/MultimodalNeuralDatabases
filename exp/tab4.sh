#! /bin/bash

TASKTYPE="perfect_ir"
QUERYTYPE="count"
STOP_ALGO_MODE="topk"
DEVICE="cuda:0"
RESULT_DIR="../results/results_processor_perfectIR"

MODEL="OFA-Sys/ofa-tiny"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.t=0.20

MODEL="OFA-Sys/ofa-base"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.t=0.20

MODEL="OFA-Sys/ofa-large"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.t=0.20

MODEL="OFA-Sys/ofa-medium"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.t=0.20

MODEL="OFA-Sys/ofa-huge"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.t=0.20

MODEL="../support-materials/models/ofa-large-ft1"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.t=0.20