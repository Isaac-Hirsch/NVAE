# NVAE with context module #

This repo provides a PyTorch implementation of NVAE extended with the `context module` from the paper *Intervening to learn and compose causally disentangled representations* ([arXiv:2507.04754](https://arxiv.org/abs/2507.04754) [stat.ML]).

NVAE ([NeurIPS 2020 Spotlight](https://arxiv.org/abs/2007.03898)) is a deep hierarchical VAE by Arash Vahdat and Jan Kautz. In this repo, the first latent code `z0` is passed through the `context module` (a `dec_conceptualizer` from [`context_module/`](context_module/)) to produce an interpretable, causally disentangled decomposition before decoding.

Dependencies, versioning, and installation are all handled by `uv`, with the included [`pyproject.toml`](./pyproject.toml) and [`uv.lock`](./uv.lock) containing all necessary information.

After [installing uv](https://docs.astral.sh/uv/getting-started/), install the project with:

```bash
uv sync
```

Then prefix any command below with `uv run` (e.g., `uv run python train.py ...`).

---

## Training

<details><summary>MNIST CM</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset concepts_mnist --batch_size 250 \
        --epochs 800 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --seed 42 \
        --arch_flag concepts --eps_dim 6 --eps_in_width 2 --eps_out_width 10 --eps_depth 5 --c_dim 6 --c_width 10
```
</details>

<details><summary>MNIST CM (fine-tune)</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
export CHECKPOINT_DIR_BASE=PATH_TO_BASE_MODEL_CHECKPOINT_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset concepts_mnist --batch_size 250 \
        --epochs 100 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --arch_flag fine-tune-concept --eps_dim 6 --eps_in_width 2 --eps_out_width 10 --eps_depth 5 --c_dim 6 --c_width 10 \
        --finetune_pt $CHECKPOINT_DIR_BASE
```
</details>

<details><summary>MNIST CM (end-to-end)</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
export CHECKPOINT_DIR_BASE=PATH_TO_BASE_MODEL_CHECKPOINT_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset concepts_mnist --batch_size 250 \
        --epochs 100 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --arch_flag fine-tune-concept-unfreeze --eps_dim 6 --eps_in_width 2 --eps_out_width 10 --eps_depth 5 --c_dim 6 --c_width 10 \
        --finetune_pt $CHECKPOINT_DIR_BASE
```
</details>

<details><summary>MNIST Base</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset concepts_mnist_obs --batch_size 250 \
        --epochs 800 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --seed 42
```

</details>

<details><summary>MNIST Base Pooled</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset concepts_mnist --batch_size 250 \
        --epochs 800 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --seed 42

```

</details>

</details>

<details><summary>MNIST CM Pooled</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR --root $CHECKPOINT_DIR --save $EXPR_ID --dataset concepts_mnist --batch_size 250 \
        --epochs 800 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --seed 42 \
        --arch_flag single-pooled-concept --eps_dim 6 --eps_in_width 2 --eps_out_width 10 --eps_depth 5 --c_dim 6 --c_width 10

```

</details>

<details><summary>3DIdent CM (fine-tune)</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
export CHECKPOINT_DIR_BASE=PATH_TO_BASE_MODEL_CHECKPOINT_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR/3DIdent_Concepts --root $CHECKPOINT_DIR --save $EXPR_ID --dataset 3DIdent_concepts-64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 40 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 10 \
        --batch_size 128 --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --arch_flag fine-tune-concept --eps_dim 3 --eps_in_width 7 --eps_out_width 50 --eps_depth 6 --c_dim 3 --c_width 50 \
        --finetune_pt $CHECKPOINT_DIR_BASE
```

</details>

<details><summary>3DIdent CM (end-to-end)</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
export CHECKPOINT_DIR_BASE=PATH_TO_BASE_MODEL_CHECKPOINT_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR/3DIdent_Concepts --root $CHECKPOINT_DIR --save $EXPR_ID --dataset 3DIdent_concepts-64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 40 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 10 \
        --batch_size 128 --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --arch_flag fine-tune-concept --eps_dim 3 --eps_in_width 7 --eps_out_width 50 --eps_depth 6 --c_dim 3 --c_width 50 \
        --finetune_pt $CHECKPOINT_DIR_BASE --freeze_iters 2000
```

</details>

<details><summary>3DIdent CM</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR/3DIdent_Concepts --root $CHECKPOINT_DIR --save $EXPR_ID --dataset 3DIdent_concepts-64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 120 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 10 \
        --batch_size 128 --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --seed 42 \
        --arch_flag concepts --eps_dim 3 --eps_in_width 7 --eps_out_width 50 --eps_depth 6 --c_dim 3 --c_width 50
```

</details>

<details><summary>3DIdent Base</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR/3DIdent_Concepts --root $CHECKPOINT_DIR --save $EXPR_ID --dataset 3DIdent_concepts_obs-64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 120 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 10 \
        --batch_size 128 --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --seed 42
```

</details>

<details><summary>3DIdent Base Pooled</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR/3DIdent_Concepts --root $CHECKPOINT_DIR --save $EXPR_ID --dataset 3DIdent_concepts-64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 120 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 10 \
        --batch_size 128 --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --seed 42
```

</details>

<details><summary>3DIdent CM Pooled</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

cd $CODE_DIR

uv run train.py --data $DATA_DIR/3DIdent_Concepts --root $CHECKPOINT_DIR --save $EXPR_ID --dataset 3DIdent_concepts-64 \
        --num_channels_enc 64 --num_channels_dec 64 --epochs 120 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales 3 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_groups_per_scale 10 \
        --batch_size 128 --ada_groups --num_process_per_node 8 --use_se --res_dist --micro_batches 2 \
        --seed 42 \
        --arch_flag single-pooled-concept --eps_dim 3 --eps_in_width 7 --eps_out_width 50 --eps_depth 6 --c_dim 3 --c_width 50
```

</details>

---

## Inference

<details><summary>MNIST</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

export IMG_DIR=PATH_TO_IMG_DIR
export ID_DIR=PATH_TO_IN_DISTRIBUTION_DATA
export DATASET=concepts_mnist

export temp=0.8
export temp_str=$(echo $temp | tr . _)
export batch_size=512

cd $CODE_DIR

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample_combo --temp=$temp --readjust_bn \
    --save "${IMG_DIR}/sample_combo_${temp_str}" --world_size 1 --local_rank 0 --batch_size $batch_size

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=compare_2_concepts_labeled --temp=$temp --readjust_bn \
    --save "${IMG_DIR}/sample_compo_compare_${temp_str}" --world_size 1 --local_rank 0 --batch_size $batch_size

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample --temp=$temp --readjust_bn \
    --save "${IMG_DIR}/sample_${temp_str}" --world_size 1 --local_rank 0 --batch_size $batch_size

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=reconstruction_test --readjust_bn\
    --save "${IMG_DIR}/reconstruction_test" --world_size 1 --local_rank 0 --data $DATA_DIR --batch_size $batch_size

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=id_metrics --readjust_bn\
    --save $IMG_DIR --world_size 1 --local_rank 0 --data $ID_DIR --batch_size $batch_size--dataset $DATASET
```

</details>

<details><summary>3DIdent</summary>

```bash
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR

export IMG_DIR=PATH_TO_IMG_DIR
export ID_DIR=PATH_TO_IN_DISTRIBUTION_DATA
export OOD_DIR=PATH_TO_OUT_OF_DISTRIBUTION_DATA
export DATASET=concepts_mnist

export temp=0.7
export temp_str=$(echo $temp | tr . _)
export batch_size=128

cd $CODE_DIR

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample_combo --temp=$temp --readjust_bn \
    --save "${IMG_DIR}/sample_combo_${temp_str}" --world_size 1 --local_rank 0 --batch_size $batch_size

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=compare_2_concepts --temp=$temp --readjust_bn \
    --save "${IMG_DIR}/sample_combo_compare_${temp_str}" --world_size 1 --local_rank 0 --batch_size $batch_size

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample --temp=$temp --readjust_bn \
    --save "${IMG_DIR}/sample_${temp_str}" --world_size 1 --local_rank 0 --batch_size $batch_size

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=reconstruction_test --readjust_bn\
    --save "${IMG_DIR}/reconstruction_test" --world_size 1 --local_rank 0 --data $DATA_DIR --batch_size $batch_size --dataset $DATASET 

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=id_metrics --readjust_bn\
    --save $IMG_DIR --world_size 1 --local_rank 0 --data $ID_DIR --batch_size $batch_size --dataset $DATASET

uv run evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=ood_metrics --readjust_bn\
    --save $IMG_DIR --world_size 1 --local_rank 0 --data $OOD_DIR --batch_size $batch_size
```

</details>

---

## Monitoring training

```bash
tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/
```

---

# Citing #

If you use this code, please cite both the context module paper and the original NVAE paper:

```bibtex
@InProceedings{markham2026,
  title = 	     {Intervening to learn and compose causally disentangled representations},
  author =       {Markham, Alex and Hirsch, Isaac and Chang, Jeri A. and Solus, Liam and Aragam, Bryon},
  booktitle = 	 {Proceedings of the Fifth Conference on Causal Learning and Reasoning},
  year = 	     {2026},
  editor = 	     {Bijan Mazaheri and Niels Richard Hansen},
  series = 	     {Proceedings of Machine Learning Research},
  month = 	     {Apr},
  publisher =    {PMLR},
}

@inproceedings{vahdat2020NVAE,
  title  = {{NVAE}: A Deep Hierarchical Variational Autoencoder},
  author = {Vahdat, Arash and Kautz, Jan},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year   = {2020},
}
```

---

# Contact #

Feel free to [raise an issue](https://github.com/Alex-Markham/context-module/issues/new/choose) or [email Alex](mailto:alex.markham@causal.dev) with questions about the context module, or reach out to [Isaac](mailto:isaachirsch1229@gmail.com) with questions specific to this NVAE integration.
