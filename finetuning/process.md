# Fine-tuning

1. Put unlabeled source images into `finetuning/to_be_labled/images/`.
2. Label them with the classes from `finetuning/to_be_labled/labels.txt`.
3. Or generate initial pseudo-labels by editing the constants in `finetuning/data_collection/prepare_dataset.py` and then running `python finetuning/data_collection/prepare_dataset.py`. It now also scans `finetuning/dataset/images/{train,val}` and infers labels for any image that does not already have a matching `.txt` file in `finetuning/dataset/labels/{train,val}`.
4. If you want to reshuffle the existing dataset and rebuild the `train`/`val` split using the configured ratio in `finetuning/data_collection/reshuffle_dataset.py`, run `python finetuning/data_collection/reshuffle_dataset.py`.
5. Review and fix the generated labels in `finetuning/dataset/labels/{train,val}`.
6. `finetuning/training/train_config.yaml` is now tuned for an RTX 4070 Ti class GPU. Reduce `batch` first if you hit CUDA out-of-memory errors, or raise it if you still have VRAM headroom.
7. Train one configuration with `python finetuning/training/train_model.py`.
8. Train all `26n`, `26s`, and `26m` checkpoints at `640`, `960`, and `1280` in ascending order with `python finetuning/training/train_all_models.py`.
