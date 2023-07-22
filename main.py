from MMNDB.Model.pipeline import Pipeline
from MMNDB.Data.data import CustomCocoDataset
from MMNDB.Utils.utils import write_report, precision_recall_f1, macro_metrics, write_report_processor
import pytorch_lightning as pl
import hydra
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(opt):

    pl.seed_everything(opt.seed)
    data = CustomCocoDataset(path=opt.data.data_path, split=opt.data.split)
    pipe = Pipeline(config=opt)

    retriver_obj_metrics = []
    processor_obj_metrics = []

    object_keys = list(data.dict_obj_id_name.keys())
    if opt.data.obj_id != 0:
        object_keys = [opt.data.obj_id]
    for obj_id in object_keys:
        processor_scores, retriever_scores = pipe.pipeline(obj_id, mode=opt.task.task_type)
        processor_obj_metrics.append(processor_scores)
        retriver_obj_metrics.append(retriever_scores)
        logging.info(f"Processor mode only, OBJECT ID: {obj_id}")
    #logger.info(f"Best threshold: {max(pipe.retriever.results_dict, key=pipe.retriever.results_dict.get)}")
    retriver_obj_metrics = list(filter(lambda x: x is not None, retriver_obj_metrics))
    processor_obj_metrics = list(filter(lambda x: x is not None, processor_obj_metrics))

    if len(retriver_obj_metrics) != 0:

        metrics_macro = precision_recall_f1(retriver_obj_metrics, mode="macro")
        metrics_micro = precision_recall_f1(retriver_obj_metrics, mode="micro")
        if opt.data.obj_id == 0:
            bootstrapped_metrics = precision_recall_f1(
                retriver_obj_metrics, mode="bootstrap"
            )
        else:
            bootstrapped_metrics = None
        write_report(
            pipe.model_name, opt, metrics_micro, metrics_macro, bootstrapped_metrics
        )

    if len(processor_obj_metrics) != 0:
        processor_macro_metrics = macro_metrics(processor_obj_metrics)
        write_report_processor(pipe.model_name, opt, processor_macro_metrics)


    logger.info(f"Broken classes: {pipe.retriever.broken_classes}")
    #logger.info(f"Best threshold: {np.mean(pipe.retriever.grid_search_value)}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
