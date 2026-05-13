# Fine-tuning

1. Put unlabeled source images into `finetuning/to_be_labled/images/`.
2. Label them with the classes from `finetuning/to_be_labled/labels.txt`.
3. Or generate initial pseudo-labels by editing the constants in `finetuning/prepare_dataset.py` and then running `python finetuning/prepare_dataset.py`.
4. Review and fix the generated labels in `finetuning/dataset/labels/{train,val}`.
5. Adjust `finetuning/train_config.yaml` for your dataset size, GPU memory, and augmentation strategy.
6. Train with `python -m ultralytics train cfg=finetuning/train_config.yaml`.
