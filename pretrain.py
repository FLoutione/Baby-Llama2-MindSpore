# nohup python /data1/cjh1/baby-llama2/Baby_Llama2/pretrain.py > /data1/cjh1/baby-llama2/Baby_Llama2/out/output1.log &

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore import jit_class
from mindspore.train.serialization import save_checkpoint
from model import Transformer, ModelArgs
from mindnlp.modules import Accumulator

from dataset import PretrainDataset
import logging

#To run with DDP on 4 gpus on 1 node, example:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py
        
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
# -----------------------------------------------------------------------------

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def forward_fn(X, Y):
    """Forward function"""
    logits = model(X, Y)
    loss = model.last_loss
    return loss / accumulate_step

def train_epoch(epoch):
    start_time=time.time()
    for step, (X, Y) in enumerate(train_loader):
        lr = get_lr(epoch*iter_per_epoch+step) if decay_lr else learning_rate
        # 改写learning_rate参数的值
        ops.assign(optimizer.learning_rate, mindspore.Tensor(lr, mindspore.float32))
        # for param_group in optimizer.param_groups:
        #     param_group['learning_rate'] = lr
        # and using the GradScaler if data type is float16
        #for micro_step in range(gradient_accumulation_steps):

        loss, grads = grad_fn(X, Y)
        # optimizer(grads)
        loss = ops.depend(loss, accumulator(grads))
        
        # with ctx:
        #     logits = model(X, Y)
        #     loss = raw_model.last_loss
        #     loss = loss / gradient_accumulation_steps

        #打印日志
        if step % log_interval == 0:
            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch,
                        max_epoch,
                        step,
                        iter_per_epoch,
                        loss.asnumpy().item(0),
                        optimizer.learning_rate.asnumpy().item(0),
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))

        # if step % save_interval == 0:
        #     model.set_train(False)
        #     save_checkpoint(model.state_dict(),'{}/baby_llama2_{}.ckpt'.format(save_dir,int(step+epoch*iter_per_epoch)))
        #     model.set_train(True)


def init_model():
    # model init
    # model init
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=64793,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )  # start with model_args from command line
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "pretrain/baby_llama2_3.ckpt")
        state_dict = mindspore.load_checkpoint(ckpt_path)
        # checkpoint_model_args = checkpoint["model_args"]
        # # force these config attributes to be equal otherwise we can't even resume training
        # # the rest of the attributes (e.g. dropout) can stay as desired from command line
        # for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        #     model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        mindspore.load_param_into_net(model, state_dict)
        # iter_num = checkpoint["iter_num"]
        # best_val_loss = checkpoint["best_val_loss"]
    return model

# I/O
if __name__=="__main__":
    out_dir = 'baby-llama2/Baby_Llama2/out'
    max_epoch = 5
    eval_interval = 1
    log_interval = 50
    save_interval = 10000
    eval_iters = 200
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    #
    gradient_accumulation_steps = 1 # used to simulate larger batch sizes
    batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
    # model 根据需要更改 
    max_seq_len = 512
    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 3e-4 # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 1000 # how many steps to warm up for
    lr_decay_iters = 80000 # should be ~= max_iters per Chinchilla
    min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # system
    dtype = 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    # exec(open("configurator.py").read())  # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    save_dir =os.path.join(out_dir , 'pretrain')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))
    # various inits, derived attributes, I/O setup
    # ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    # torch.manual_seed(1337 + seed_offset)
    # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": mindspore.float32, "float16": mindspore.float16}[dtype]
    # ctx = (
    #     nullcontext()
    #     # if device_type == "cpu"
    #     # else torch.cuda.amp.autocast()
    # )
    #
    best_val_loss = 1e9
    #
    #-----init dataloader------
    data_path_list=[
        './baby-llama2/Baby_Llama2/data/pretrain_data.bin'
        #'./data/baidubaike_563w.bin',
        #'./data/medical_book.bin',
        # './data/medical_encyclopedia.bin',
        # './data/medical_qa.bin',
        # './baby-llama2/Baby_Llama2/data/wiki.bin'
    ]
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len,memmap=True)
    # train_sampler = mindspore.dataset.DistributedSampler(num_shards=None, shard_id=None)
    train_loader = mindspore.dataset.GeneratorDataset(
        train_ds,
        column_names=["input", "label"],
        shuffle=False,        
        num_parallel_workers=0 if os.name == 'nt' else 4,
        # sampler=train_ds
    )
    train_loader = train_loader.batch(batch_size, drop_remainder=True)

    #init model
    model=init_model()
    model.set_train(True)
    for key, param in model.parameters_and_names():
        print(param.name, param.shape)
    print('number of model parameters: {}'.format(model.parameters_and_names()))

    iter_per_epoch=len(train_loader)
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, beta1, beta2)
    accumulate_step = 1
    accumulator = Accumulator(optimizer, accumulate_step)
    
    # 梯度函数
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)

    for epoch in range(max_epoch):
        train_epoch(epoch)
        save_checkpoint(model,'{}/baiduwike_llama2_{}.ckpt'.format(save_dir, epoch))