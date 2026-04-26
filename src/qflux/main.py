import logging

import diffusers
import transformers

from qflux.data.config import Config, TrMode
from qflux.data.dataset import loader
from qflux.utils.logger import get_logger
from qflux.utils.options import parse_args
from qflux.utils.seed import seed_everything


logger = get_logger(__name__, log_level="INFO")


def import_trainer(config: Config):
    trainer_type = config.trainer_type
    if trainer_type == "QwenImageEdit":
        from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer

        return QwenImageEditTrainer
    elif trainer_type == "FluxKontext":
        from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer

        return FluxKontextLoraTrainer
    elif trainer_type == "QwenImageEditPlus":
        from qflux.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer

        return QwenImageEditPlusTrainer
    elif trainer_type == "DreamOmni2":
        from qflux.trainer.dreamomni2_trainer import DreamOmni2Trainer

        return DreamOmni2Trainer
    else:
        raise ValueError(f"Invalid trainer type: {trainer_type}")


def main():
    """使用模块化的 Trainer 类进行训练"""
    # 解析配置
    fmt = "%(asctime)s %(levelname)s [pid=%(process)d] %(filename)s:%(lineno)d %(funcName)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    config = parse_args()

    # 设置日志级别
    if hasattr(config, "local_rank") and config.local_rank == 0:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    seed_everything(1234)

    # 创建训练器
    Trainer = import_trainer(config)
    trainer = Trainer(config)
    if config.mode != TrMode.fit:
        # if not in training, skip caption dropout
        config.data.init_args.caption_dropout_rate = 0

    # 加载数据
    batch_size = config.data.batch_size
    shuffle = config.data.shuffle
    droplast = True
    if config.mode == TrMode.cache:
        batch_size = 1
        shuffle = False
        droplast = False
        # Disable cache loading during cache building to avoid loading incomplete/corrupted cache
        config.data.init_args.use_cache = False
        logging.info("In cache mode, adjust batch_size, shuffle, droplast")
        logging.info("\tbatch_size {batch_size}")
        logging.info(f"\tshuffle {shuffle}")
        logging.info(f"\tdroplast {droplast}")
        logging.info(f"\tuse_cache disabled during cache building")
    train_dataloader = loader(
        config.data.class_path,
        config.data.init_args,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        shuffle=shuffle,
        drop_last=droplast,
    )

    # Build val DataLoader when validation is enabled and has a dataset config
    val_dataloader = None
    if (
        config.validation.enabled
        and hasattr(config.validation, "dataset")
        and config.validation.dataset is not None
    ):
        val_init_args = config.validation.dataset.init_args
        if config.mode == TrMode.cache:
            val_init_args.use_cache = False
        val_dataloader = loader(
            config.validation.dataset.class_path,
            val_init_args,
            batch_size=batch_size if config.mode != TrMode.cache else 1,
            num_workers=config.data.num_workers,
            shuffle=False,
            drop_last=False,
        )

    if config.mode == TrMode.cache:
        try:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(trainer.text_encoder)
        except Exception as e:
            print("remove_hook skipped:", repr(e))

        trainer.cache(train_dataloader, val_dataloader=val_dataloader)
    else:
        trainer.fit(train_dataloader, val_dataloader=val_dataloader)


if __name__ == "__main__":
    main()
