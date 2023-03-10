python -m torch.distributed.launch --nproc_per_node=8 --master_port=11223 main.py --batch_size=2 --num_steps=100000 --lrdecay --eval_num=500 --lr=6e-6 --decay=0.1 --norm_pix_loss
