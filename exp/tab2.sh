
QUERYTYPE="count"
STOP_ALGO_MODE="mixed"
DEVICE="cuda:0"
RESULT_DIR="../results/tab2"
CLIP_MODEL="ViT-L/14@336px"
STOP_ALGO="../support-materials/stop-algo/ViT-L14@336px_StopAlgo.pt"
PROCESSOR_MODEL="../support-materials/models/ofa-large-ft1"
PROCESSOR_BATCH_SIZE=42



TASKTYPE="full_pipeline"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$PROCESSOR_MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10 retriever.clip_model=$CLIP_MODEL retriever.stop_algo_path=$STOP_ALGO processor.batch_size=$PROCESSOR_BATCH_SIZE

TASKTYPE="perfect_ir"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$PROCESSOR_MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10 retriever.clip_model=$CLIP_MODEL retriever.stop_algo_path=$STOP_ALGO processor.batch_size=$PROCESSOR_BATCH_SIZE

TASKTYPE="noisy_ir"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$PROCESSOR_MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10 retriever.clip_model=$CLIP_MODEL retriever.stop_algo_path=$STOP_ALGO processor.batch_size=$PROCESSOR_BATCH_SIZE

TASKTYPE="damaging_ir"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$PROCESSOR_MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10 retriever.clip_model=$CLIP_MODEL retriever.stop_algo_path=$STOP_ALGO processor.batch_size=$PROCESSOR_BATCH_SIZE

#ORIGINAL MODEL (NO FT)

TASKTYPE="full_pipeline"
PROCESSOR_MODEL="OFA-Sys/ofa-large"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$PROCESSOR_MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10 retriever.clip_model=$CLIP_MODEL retriever.stop_algo_path=$STOP_ALGO processor.batch_size=$PROCESSOR_BATCH_SIZE

TASKTYPE="perfect_ir"
PROCESSOR_MODEL="OFA-Sys/ofa-large"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$PROCESSOR_MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10 retriever.clip_model=$CLIP_MODEL retriever.stop_algo_path=$STOP_ALGO processor.batch_size=$PROCESSOR_BATCH_SIZE

TASKTYPE="noisy_ir"
PROCESSOR_MODEL="OFA-Sys/ofa-large"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$PROCESSOR_MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10 retriever.clip_model=$CLIP_MODEL retriever.stop_algo_path=$STOP_ALGO processor.batch_size=$PROCESSOR_BATCH_SIZE

TASKTYPE="damaging_ir"
PROCESSOR_MODEL="OFA-Sys/ofa-large"
python ../main.py task.task_type=$TASKTYPE task.query_type=$QUERYTYPE processor.checkpoint_processor=$PROCESSOR_MODEL experiments.results_path=$RESULT_DIR device=$DEVICE retriever.stop_algo_type=$STOP_ALGO_MODE retriever.k=10 retriever.clip_model=$CLIP_MODEL retriever.stop_algo_path=$STOP_ALGO processor.batch_size=$PROCESSOR_BATCH_SIZE
