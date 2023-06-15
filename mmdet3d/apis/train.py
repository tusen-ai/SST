from mmdet.apis import train_detector
from .seq_training_apis import train_detector_seq
from mmseg.apis import train_segmentor


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    use_seq_code = cfg.data.get('use_seq_code', False)
    weak_shuffle_cfg = cfg.data.get('weak_shuffle_cfg', None)
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    elif use_seq_code:
        train_detector_seq(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta,
            weak_shuffle_cfg=weak_shuffle_cfg,
        )
    else:
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
