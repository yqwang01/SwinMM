# Note for SwinMM Finetuning

## Setup

```bash
conda activate <env_name>
bash ./scripts/setup_env.sh
```

## Training

### Prerequisite

Launch in memory database, only need once

```bash
# It launches redis at ports 39996-39999.
bash ./scripts/start_redis.sh
```

**NOTE**

- If **data or preprocessing** changed, run `pkill redis-server` before further experiments
- Try `--workers` from 8 to 32 for best performance
- First epoch after launch the server maybe slow, should be fast later
- Set `--redis_ports <ports>` according to your redis setup.


```bash
python main.py
--feature_size=48
--batch_size=1
--logdir="swin_mm_test/"
--roi_x=64
--roi_y=64
--roi_z=64
--optim_lr=1e-4
--lrschedule="warmup_cosine"
--infer_overlap=0.5
--save_checkpoint
--data_dir="/dataset/dataset0/"
--distributed
--use_ssl_pretrained
--pretrained_dir="./pretrained_models/"
--pretrained_model_name="model_bestValRMSE.pt"
```

## Testing


```bash
python test.py
--feature_size=48
--batch_size=1
--exp_name="swin_mm_test/"
--roi_x=64
--roi_y=64
--roi_z=64
--infer_overlap=0.5
--data_dir="/dataset/dataset0/"
--pretrained_dir="./runs/multiview_101021/"
--pretrained_model_name="model.pt"
```
