import logging
import os
from scipy.stats import bootstrap
import numpy as np

logger = logging.getLogger(__name__)


def write_report(
    file_name,
    opt,
    micro_metrics: dict,
    macro_metrics: dict,
    bootstrap_metrics: dict = None,
):
    logger.info("Writing report")
    path = opt.experiments.results_path
    
    if opt.retriever.stop_algo_type == "threshold":
        header_k = "threshold"
        k = opt.retriever.t
    else:
        header_k = "k"
        k = opt.retriever.k

    if not os.path.isdir(path):
        os.mkdir(path)
    model_res = os.path.join(path, file_name + ".csv")

    if bootstrap_metrics is None:
        header = f"{header_k}, micro_f1, micro_precision, micro_recall, macro_f1, macro_precision, macro_recall, percentage_RT, percentage_RT_std\n"
    else:
        header = f"{header_k}, micro_f1, micro_f1_ci_low, micro_f1_ci_high, micro_f1_std_error, micro_precision, micro_precision_ci_low, micro_precision_ci_high, micro_precision_std_error, micro_recall, micro_recall_ci_low, micro_recall_ci_high, micro_recall_std_error, macro_f1, macro_f1_ci_low, macro_f1_ci_high, macro_f1_std_error, macro_precision, macro_precision_ci_low, macro_precision_ci_high, macro_precision_std_error, macro_recall, macro_recall_ci_low, macro_recall_ci_high, macro_recall_std_error, percentage_RT, percentage_RT_std\n"
    # check if file exists
    if not os.path.isfile(model_res):
        with open(model_res, "w") as f:
            f.write(header)
    
    if 'percentage_RT_std_error' not in macro_metrics:
        macro_metrics['percentage_RT_std_error'] = 0
    
    if 'percentage_RT' not in micro_metrics:
        micro_metrics['percentage_RT'] = 0

    with open(model_res, "a") as f:
        if bootstrap_metrics is None:
            f.write(
                f"{k}, {micro_metrics['f1']}, {micro_metrics['precision']}, {micro_metrics['recall']}, {macro_metrics['f1']}, {macro_metrics['precision']}, {macro_metrics['recall']}, {macro_metrics['percentage_RT']}, {macro_metrics['percentage_RT_std_error']}\n"
            )
        else:
            f.write(
                f"{k}, {micro_metrics['f1']}, {bootstrap_metrics['micro_f1_ci_low']}, {bootstrap_metrics['micro_f1_ci_high']}, {bootstrap_metrics['micro_f1_std_error']}, {micro_metrics['precision']}, {bootstrap_metrics['micro_precision_ci_low']}, {bootstrap_metrics['micro_precision_ci_high']}, {bootstrap_metrics['micro_precision_std_error']}, {micro_metrics['recall']}, {bootstrap_metrics['micro_recall_ci_low']}, {bootstrap_metrics['micro_recall_ci_high']}, {bootstrap_metrics['micro_recall_std_error']}, {macro_metrics['f1']}, {bootstrap_metrics['macro_f1_ci_low']}, {bootstrap_metrics['macro_f1_ci_high']}, {bootstrap_metrics['macro_f1_std_error']}, {macro_metrics['precision']}, {bootstrap_metrics['macro_precision_ci_low']}, {bootstrap_metrics['macro_precision_ci_high']}, {bootstrap_metrics['macro_precision_std_error']}, {macro_metrics['recall']}, {bootstrap_metrics['macro_recall_ci_low']}, {bootstrap_metrics['macro_recall_ci_high']}, {bootstrap_metrics['macro_recall_std_error']}, {macro_metrics['percentage_RT']}, {macro_metrics['percentage_RT_std_error']}\n"
            )
    logger.info("Report written")


def precision_recall_f1(metrics_list, mode="macro"):
    if mode == "macro":
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_percentange = 0
        for metric in metrics_list:
            precison = metric["precision"]
            recall = metric["recall"]
            f1 = metric["f1"]
            total_precision += precison
            total_recall += recall
            total_f1 += f1
            total_percentange  += metric["percentage_RT"]
        return {
            "precision": total_precision / len(metrics_list),
            "recall": total_recall / len(metrics_list),
            "f1": total_f1 / len(metrics_list),
            "percentage_RT": total_percentange / len(metrics_list)
        }
    elif mode == "micro":
        total_tp = 0
        total_fn = 0
        total_fp = 0
        for metric in metrics_list:
            tp = metric["tp"]
            fn = metric["fn"]
            fp = metric["fp"]
            total_tp += tp
            total_fn += fn
            total_fp += fp
        if (2 * total_tp + total_fp + total_fn) == 0:
            f1 = 0
        else:
            f1 = (2 * (total_tp)) / (2 * total_tp + total_fp + total_fn)
        if total_tp + total_fp == 0:
            precision = 0
        else:
            precision = total_tp / (total_tp + total_fp)
        if total_tp + total_fn == 0:
            recall = 0
        else:
            recall = total_tp / (total_tp + total_fn)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    elif mode == "bootstrap":
        precison_list = np.asarray([i["precision"] for i in metrics_list])
        recall_list = np.asarray([i["recall"] for i in metrics_list])
        f1_list = np.asarray([i["f1"] for i in metrics_list])
        percentage_RT = np.asarray([i["percentage_RT"] for i in metrics_list])

        precision_ci = bootstrap((precison_list,), np.mean)
        recall_ci = bootstrap((recall_list,), np.mean)
        f1_ci = bootstrap((f1_list,), np.mean)
        percentage_ci = bootstrap((percentage_RT,), np.mean)

        tp_sum = np.sum([i["tp"] for i in metrics_list])
        fp_sum = np.sum([i["fp"] for i in metrics_list])
        fn_sum = np.sum([i["fn"] for i in metrics_list])

        precision_np = np.asarray(
            [True for _ in range(tp_sum)] + [False for _ in range(fp_sum)]
        )
        recall_np = np.asarray([True for _ in range(tp_sum)] + [False for _ in range(fn_sum)])
        f1_np = np.asarray(
            [True for _ in range(2 * tp_sum)] + [False for _ in range(fp_sum + fn_sum)]
        )

        micro_precision_ci = bootstrap((precision_np,), np.mean)
        micro_recall_ci = bootstrap((recall_np,), np.mean)
        micro_f1_ci = bootstrap((f1_np,), np.mean)

        return {
            "macro_precision_std_error": precision_ci.standard_error,
            "macro_precision_ci_low": precision_ci.confidence_interval.low,
            "macro_precision_ci_high": precision_ci.confidence_interval.high,
            "macro_recall_std_error": recall_ci.standard_error,
            "macro_recall_ci_low": recall_ci.confidence_interval.low,
            "macro_recall_ci_high": recall_ci.confidence_interval.high,
            "macro_f1_std_error": f1_ci.standard_error,
            "macro_f1_ci_low": f1_ci.confidence_interval.low,
            "macro_f1_ci_high": f1_ci.confidence_interval.high,
            "micro_precision_std_error": micro_precision_ci.standard_error,
            "micro_precision_ci_low": micro_precision_ci.confidence_interval.low,
            "micro_precision_ci_high": micro_precision_ci.confidence_interval.high,
            "micro_recall_std_error": micro_recall_ci.standard_error,
            "micro_recall_ci_low": micro_recall_ci.confidence_interval.low,
            "micro_recall_ci_high": micro_recall_ci.confidence_interval.high,
            "micro_f1_std_error": micro_f1_ci.standard_error,
            "micro_f1_ci_low": micro_f1_ci.confidence_interval.low,
            "micro_f1_ci_high": micro_f1_ci.confidence_interval.high,
            "percentage_RT_std_error": percentage_ci.standard_error,
        }
    elif mode == "weighted":
        pass
    else:
        logger.error("Invalid mode for calculating precision_recall_f1")


def macro_metrics(metrics_list):
    total_exact_match_acc_tp = 0
    total_exact_match_acc_fp = 0
    total_accuracy = 0
    total_delta_error_tp = 0
    total_delta_error_fp = 0
    total_delta_error = 0
    total_total_error_tp = 0
    total_total_error_fp = 0
    total_total_error_fn = 0
    total_total_error = 0

    for metric in metrics_list:
  
        total_exact_match_acc_tp += metric["exact_match_acc_tp"]
        total_accuracy += metric["exact_match_acc"]
        total_exact_match_acc_fp += metric["exact_match_acc_fp"]
        total_delta_error_tp += metric["delta_error_tp"]
        total_delta_error_fp += metric["delta_error_fp"]
        total_delta_error += metric["total_delta_error"]
        total_total_error_tp += metric["total_error_tp"]
        total_total_error_fp += metric["total_error_fp"]
        total_total_error_fn += metric["total_error_fn"]
        total_total_error += metric["total_error"]
    
    final_exact_match_acc_fp = total_exact_match_acc_fp / len(metrics_list)
    final_exact_match_acc_tp = total_exact_match_acc_tp / len(metrics_list)
    final_delta_error_fp = total_delta_error_fp / len(metrics_list)
    final_delta_error_tp = total_delta_error_tp / len(metrics_list)
    final_total_error_fp = total_total_error_fp / len(metrics_list)
    final_total_error_tp = total_total_error_tp / len(metrics_list)
    final_total_error_fn = total_total_error_fn / len(metrics_list)
    final_total_error = total_total_error / len(metrics_list)
    final_delta_error = total_delta_error / len(metrics_list)
    final_accuracy = total_accuracy / len(metrics_list)



    total_error_list = [i["total_error"] for i in metrics_list]
    total_error_fp_list = [i["total_error_fp"] for i in metrics_list]
    total_error_tp_list = [i["total_error_tp"] for i in metrics_list]
    total_error_fn_list = [i["total_error_fn"] for i in metrics_list]
    delta_error_list = [i["total_delta_error"] for i in metrics_list]
    delta_error_fp_list = [i["delta_error_fp"] for i in metrics_list]
    delta_error_tp_list = [i["delta_error_tp"] for i in metrics_list]
    exact_match_acc_list = [i["exact_match_acc"] for i in metrics_list]
    exact_match_acc_tp_list = [i["exact_match_acc_tp"] for i in metrics_list]
    exact_match_acc_fp_list = [i["exact_match_acc_fp"] for i in metrics_list]


    
    
    total_error_ci = bootstrap((total_error_list,), np.mean)
    total_error_fp_ci = bootstrap((total_error_fp_list,), np.mean)
    total_error_tp_ci = bootstrap((total_error_tp_list,), np.mean)
    total_error_fn_ci = bootstrap((total_error_fn_list,), np.mean)
    delta_error_ci = bootstrap((delta_error_list,), np.mean)
    delta_error_fp_ci = bootstrap((delta_error_fp_list,), np.mean)
    delta_error_tp_ci = bootstrap((delta_error_tp_list,), np.mean)
    exact_match_acc_ci = bootstrap((exact_match_acc_list,), np.mean)
    exact_match_acc_tp_ci = bootstrap((exact_match_acc_tp_list,), np.mean)
    exact_match_acc_fp_ci = bootstrap((exact_match_acc_fp_list,), np.mean)



    return {
        "total_error": final_total_error,
        "total_error_std_error": total_error_ci.standard_error,
        "total_error_tp": final_total_error_tp,
        "total_error_tp_std_error": total_error_tp_ci.standard_error,
        "total_error_fp": final_total_error_fp,
        "total_error_fp_std_error": total_error_fp_ci.standard_error,
        "total_error_fn": final_total_error_fn,
        "total_error_fn_std_error": total_error_fn_ci.standard_error,
        "delta_error": final_delta_error,
        "delta_error_std_error": delta_error_ci.standard_error,
        "delta_error_tp": final_delta_error_tp,
        "delta_error_tp_std_error": delta_error_tp_ci.standard_error,
        "delta_error_fp": final_delta_error_fp,
        "delta_error_fp_std_error": delta_error_fp_ci.standard_error,
        "exact_match_acc": final_accuracy,
        "exact_match_acc_std_error": exact_match_acc_ci.standard_error,
        "exact_match_acc_tp": final_exact_match_acc_tp,
        "exact_match_acc_tp_std_error": exact_match_acc_tp_ci.standard_error,
        "exact_match_acc_fp": final_exact_match_acc_fp,
        "exact_match_acc_fp_std_error": exact_match_acc_fp_ci.standard_error
        }


def write_report_processor(file_name,
    opt,
    macro_metrics: dict
):
    logger.info("Writing report Processor")
    path = opt.experiments.results_path
    
    header_k = "n_noise"
    k = opt.retriever.noisy_ir_noise

    if not os.path.isdir(path):
        os.mkdir(path)
    model_res = os.path.join(path, f"proc_{file_name}_{opt.processor.checkpoint_processor.replace('/','')}_{opt.task.query_type}_{opt.task.task_type}.csv")

    header = f"{header_k}, total_error, total_erro_std, total_error_tp, total_error_tp_std, total_error_fp, total_error_fp_std, total_error_fn, total_error_fn_std, delta_error, delta_error_std, delta_error_tp, delta_error_tp_std, delta_error_fp, delta_error_fp_std, exact_match_acc, exact_match_acc_std, exact_match_acc_tp, exact_match_acc_tp_std, exact_match_acc_fp, exact_match_acc_fp_std \n"
    
    if not os.path.isfile(model_res):
        with open(model_res, "w") as f:
            f.write(header)

    with open(model_res, "a") as f:
        f.write(
                f"{k}, {macro_metrics['total_error']}, {macro_metrics['total_error_std_error']}, {macro_metrics['total_error_tp']}, {macro_metrics['total_error_tp_std_error']}, {macro_metrics['total_error_fp']}, {macro_metrics['total_error_fp_std_error']}, {macro_metrics['total_error_fn']}, {macro_metrics['total_error_fn_std_error']}, {macro_metrics['delta_error']}, {macro_metrics['delta_error_std_error']}, {macro_metrics['delta_error_tp']}, {macro_metrics['delta_error_tp_std_error']}, {macro_metrics['delta_error_fp']}, {macro_metrics['delta_error_fp_std_error']}, {macro_metrics['exact_match_acc']}, {macro_metrics['exact_match_acc_std_error']}, {macro_metrics['exact_match_acc_tp']}, {macro_metrics['exact_match_acc_tp_std_error']}, {macro_metrics['exact_match_acc_fp']}, {macro_metrics['exact_match_acc_fp_std_error']} \n"
        )
    logger.info("Report written")