from verl.trainer.distillation.losses import distillation_loss
from verl.utils.metric import AggregationType, Metric


def skill_opsd_loss(config, distillation_config, model_output, data, dp_group=None):
    loss, metrics = distillation_loss(config, distillation_config, model_output, data)
    metrics["distillation/loss"] = Metric(value=loss, aggregation=AggregationType.SUM)
    return loss, metrics
