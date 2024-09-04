# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco128.yaml imgsz=640 epochs=100 batch=16
"""

import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
)


class BaseTrainer:
    """
    BaseTrainer.

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in ("cpu", "mps"):
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        #使用 check_model_file_from_stem 函数从 self.args.model 参数中加载模型文件
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        try:
            #如果 self.args.task 是 "classify"（分类任务），则调用 check_cls_dataset 函数来处理数据集，该函数预期接收一个分类任务的数据集路径或配置
            if self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data)
            #如果 self.args.data 参数的值以 ".yaml" 或 ".yml" 结尾，
            # 或者 self.args.task 是 "detect"（检测）、"segment"（分割）、"pose"（姿态估计）、"obb"（定向边界框）中的任何一个，
            # 则调用 check_det_dataset 函数来处理数据集
            elif self.args.data.split(".")[-1] in ("yaml", "yml") or self.args.task in (
                "detect",
                "segment",
                "pose",
                "obb",
            ):
                #读取数据集配置文件，获取数据集相应配置
                self.data = check_det_dataset(self.args.data)
                #如果 check_det_dataset 函数返回的数据集中包含 "yaml_file" 键，则将该键的值（即 YAML 文件的路径）赋值给 self.args.data
                if "yaml_file" in self.data:
                    self.args.data = self.data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch == -1:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        #任何层名中包含".dfl"的层都将被冻结
        always_freeze_names = [".dfl"]  # always freeze these layers
        #在每个元素前添加"model."前缀来生成完整的层名.  将这个列表与always_freeze_names合并
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        #遍历模型中的所有参数（包括权重和偏置），其中k是参数的名称，v是参数本身
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                #将该参数的requires_grad属性设置为False，表示在训练过程中不会更新该参数的权重
                v.requires_grad = False
            #如果一个浮点类型的参数被冻结了，则放开冻结。（可能是因为浮点参数必须得更新）
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        #尝试从self.args.amp（可能是一个布尔值或字符串，表示是否启用自动混合精度）创建一个PyTorch张量self.amp，并将其移动到指定的设备（如GPU）上
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        #在单GPU或分布式数据并行（DDP）模式下
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            #函数来检查模型是否支持AMP
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            #备份默认的回调函数，因为check_amp可能会重置它们，之后再恢复。
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        #在DDP模式下
        if RANK > -1 and world_size > 1:  # DDP
            #将self.amp张量从rank 0广播到其他所有rank，以确保所有进程中的AMP设置一致。
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        #使用torch.cuda.amp.GradScaler创建一个梯度缩放器self.scaler，其启用状态由self.amp决定。梯度缩放器用于在AMP训练过程中自动调整梯度的缩放，以减少数值不稳定和溢出
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        #如果是在多GPU环境下（world_size > 1），则将模型包装在nn.parallel.DistributedDataParallel中以支持分布式数据并行训练。这里指定了每个进程使用的设备ID为[RANK]
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])

        # Check imgsz
        #计算网格大小gs，这是模型最大步长（如果有的话）和32中的较大值。这个值用于后续的多尺度训练
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        #使用check_imgsz函数根据网格大小gs调整self.args.imgsz（输入图像的尺寸），以确保它符合特定的要求（如步长的倍数）
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            #估计最佳的批量大小。这个函数会考虑模型、图像尺寸和AMP设置来确定一个合适的批量大小。
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        #如果是分布式训练，则批量大小会被当前世界大小world_size整除，以确保每个GPU上的批量大小是均匀的
        batch_size = self.batch_size // max(world_size, 1)
        #生成训练加载器
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")
        #如果是在主进程（RANK in (-1, 0)）中
        if RANK in (-1, 0):
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            #对于测试集，如果任务类型（self.args.task）是"obb"（可能是指Oriented Bounding Box，即带方向的边界框），则保持批量大小不变；否则，将批量大小加倍
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            #如果是在主进程（RANK in (-1, 0)）中，则初始化验证器self.validator，并设置用于跟踪训练和验证过程中各种指标的字典self.metrics
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            #使用ModelEMA（模型指数移动平均）来平滑模型参数，这有助于在训练过程中获得更稳定的模型
            self.ema = ModelEMA(self.model)
            #如果启用了绘图功能（self.args.plots），则调用plot_training_labels方法来绘制训练标签的图表
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        #计算梯度累积的步数self.accumulate，这是为了在更新模型参数之前累积足够的损失，以模拟更大的批量大小。
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        #根据批量大小、梯度累积步数和总训练步数（self.args.nbs）来调整权重衰减（weight_decay）
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        #使用build_optimizer方法构建优化器，该方法接受模型、优化器名称、学习率、动量、权重衰减和迭代次数等参数。
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        #调用_setup_scheduler方法来设置学习率调度器
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        #当world_size > 1时，表示有多个GPU参与训练，因此需要设置分布式数据并行（DDP）环境
        if world_size > 1:
            #初始化DDP所需的所有配置。确保了模型的不同部分可以在不同的GPU上并行处理，同时保持数据一致性和梯度同步
            self._setup_ddp(world_size)
        #负责设置与训练过程相关的所有配置，无论是否启用DDP
        self._setup_train(world_size)

        #训练数据加载器（train_loader）中的批次（batch）总数
        nb = len(self.train_loader)  # number of batches
        #计算了预热（warmup）阶段的迭代次数nw。如果设置了预热周期数（self.args.warmup_epochs）大于0，则根据预热周期数和批次总数计算预热迭代次数，并确保其至少为100。如果没有设置预热周期数（即小于或等于0），则nw被设置为-1，表示不进行预热。
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        #初始化最后一个优化步骤的索引为-1，可能用于跟踪优化器的更新次数。
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        #调用训练开始前需要执行的回调函数，这些函数可能用于执行一些初始化操作，如设置日志记录器、保存模型配置等。
        self.run_callbacks("on_train_start")
        #使用LOGGER.info打印训练开始时的信息，包括图像尺寸、数据加载器的工作线程数、日志记录的位置以及训练的总时长或总轮次
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        #如果设置了self.args.close_mosaic，则计算一个特定的批次索引base_idx，这个索引对应于在训练的最后几个epoch中关闭马赛克增强（一种数据增强技术）的起始点
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            #将base_idx及其后两个索引添加到self.plot_idx列表中，可能用于后续的可视化或日志记录
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            #将模型设置为训练模式。在PyTorch中，某些层（如Dropout和BatchNorm）在训练模式和评估模式下的行为是不同的
            self.model.train()
            #如果使用了分布式训练（RANK != -1），则通过self.train_loader.sampler.set_epoch(epoch)更新数据加载器的采样器。
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            #初始化一个进度条来显示训练进度
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            #如果当前epoch是最后一个需要关闭马赛克增强的epoch
            if epoch == (self.epochs - self.args.close_mosaic):
                #关闭数据加载器中的马赛克增强
                self._close_dataloader_mosaic()
                #重置数据加载器
                self.train_loader.reset()

            #如果当前进程是主进程
            if RANK in (-1, 0):
                #打印当前epoch的进度信息
                LOGGER.info(self.progress_string())
                #使用TQDM包装器来美化进度条的显示
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            # 在每个epoch开始前，清零优化器的梯度是非常重要的。这是因为在反向传播过程中，梯度会累加到优化器的.grad属性中，
            # 如果不进行清零，那么这些梯度会在下一个epoch中继续累加，导致训练过程出错
            self.optimizer.zero_grad()
            #在一个epoch内遍历数据加载器（train_loader）中的每个批次（batch），执行前向传播、反向传播和优化步骤，并记录相关信息和日志
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                #当前迭代次数
                ni = i + nb * epoch
                #小于或等于预热迭代次数nw
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                #使用torch.cuda.amp.autocast上下文管理器自动选择最佳的数据类型和精度进行前向传播，以减少GPU内存消耗和提高计算速度
                with torch.cuda.amp.autocast(self.amp):
                    #对批次数据进行预处理，然后通过模型进行前向传播，计算损失
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    #如果使用了分布式训练（RANK != -1），则根据世界大小（world_size）调整损失值，以便在多个进程间平均
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                #使用自动混合精度（AMP）的scaler对象对损失进行缩放，然后执行反向传播
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                #如果自上次优化步骤以来已经积累了足够的梯度（ni - last_opt_step >= self.accumulate），则执行优化步骤
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    #如果设置了训练时间限制（self.args.time），则检查是否已超出训练时间，并相应地停止训练
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                # 计算并显示当前批次的GPU内存使用情况、损失值和批次大小等信息
                # 计算预留的显存大小，将单位转换为GB（torch.cuda.memory_reserved() / 1E9），如果cuda不可用则显示0
                # ':.3g'：这是一个字符串格式化选项，用于将前面的计算结果格式化为一个字符串
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                #如果当前进程是主进程（RANK in (-1, 0)），则更新进度条的描述，并调用批次结束时的回调函数
                if RANK in (-1, 0):
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            # 遍历优化器的参数组（param_groups），为每个参数组记录其学习率（lr），并将这些学习率以字典的形式存储在self.lr中。
            # 这样做是为了方便日志记录器（loggers）记录每个参数组的学习率
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            #如果当前进程是主进程
            if RANK in (-1, 0):
                #检查当前epoch是否是训练过程中的最后一个epoch
                final_epoch = epoch + 1 == self.epochs
                #使用EMA技术更新模型的某些属性。EMA通常用于平滑模型参数，以获得更稳定的模型表现。这里指定了要包括在EMA更新中的属性列表
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                #决定是否执行验证操作:
                # 是否启用了验证（self.args.val）、
                # 是否达到了验证周期（epoch+1 % self.args.val_period == 0）、
                # 是否接近训练结束（self.epochs - epoch <= 10）、
                # 是否是最终epoch、
                # 是否触发了早停条件（self.stopper.possible_stop）或
                # 是否已经设置了停止标志（self.stop）
                if (self.args.val and (((epoch+1) % self.args.val_period == 0) or (self.epochs - epoch) <= 10)) \
                    or final_epoch or self.stopper.possible_stop or self.stop:
                    #进行验证
                    self.metrics, self.fitness = self.validate()
                #将训练损失项（self.label_loss_items(self.tloss)）、验证指标（self.metrics）和学习率（self.lr）合并为一个字典，并调用self.save_metrics()方法保存这些指标
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                #根据早停条件（self.stopper(epoch + 1, self.fitness)）或是否是最终epoch来更新停止标志（self.stop）
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                #如果设置了训练时间限制（self.args.time），并且已经超过了指定的训练时间（以小时为单位），则同样设置self.stop为True
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                #如果启用了模型保存（self.args.save）或达到了最终epoch，则调用self.save_model()方法保存模型
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            #记录当前epoch的时间
            t = time.time()
            #更新self.epoch_time为当前epoch所花费的时间
            self.epoch_time = t - self.epoch_time_start
            #重置self.epoch_time_start为当前时间，以便为下一个epoch的时间记录做准备
            self.epoch_time_start = t
            #
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                #如果设置了训练时间限制
                if self.args.time:
                    #计算平均每个epoch所需的时间，并根据剩余时间动态调整总epochs数（self.epochs）
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    #设置self.scheduler.last_epoch为当前epoch，以确保学习率调度器正确地根据当前的epoch数调整学习率
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                #调用self.scheduler.step()来更新学习率
                self.scheduler.step()
            self.run_callbacks("on_fit_epoch_end")
            #调用torch.cuda.empty_cache()来清理GPU缓存
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            #如果是在分布式数据并行（DDP）训练模式下（RANK != -1）
            if RANK != -1:  # if DDP training
                #通过dist.broadcast_object_list广播self.stop标志给所有进程。这是为了确保当主进程（RANK == 0）决定停止训练时，所有其他进程也停止
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            #调用self.final_eval()方法执行最终的验证步骤。这通常涉及在验证集（或测试集）上运行模型，以评估其在未见过的数据上的性能。这可能是计算准确率、损失、或其他相关指标的过程
            self.final_eval()
            #如果通过命令行参数（self.args.plots）指定了绘图，则调用self.plot_metrics()方法来绘制训练过程中收集的指标图
            if self.args.plots:
                self.plot_metrics()
            #可能用于保存模型、清理资源、发送通知等
            self.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import pandas as pd  # scope for faster startup

        #将当前训练周期的指标（self.metrics）与一个额外的“fitness”指标合并成一个新的字典 metrics
        metrics = {**self.metrics, **{"fitness": self.fitness}}
        #使用 pandas 读取一个CSV文件（self.csv），然后将内容转换为一个字典，其中键是CSV文件的列名（去除首尾空格），值是对应列的数据列表
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()}
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": metrics,
            "train_results": results,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }

        # Save last and best
        #将检查点保存到 self.last 指定的路径，作为最新检查点
        torch.save(ckpt, self.last)
        #如果当前适应度等于最佳适应度，则将检查点也保存到 self.best 指定的路径，作为最佳模型检查点
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        #如果设置了保存周期（self.save_period），并且当前训练周期是该周期的倍数，则将检查点保存到以当前周期数命名的文件中
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f"epoch{self.epoch}.pt")

    @staticmethod
    def get_dataset(data):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        #函数检查self.model是否已经是torch.nn.Module的实例
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt["model"].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in "imgsz", "batch", "device":  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt["epoch"] + 1
        if ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer
