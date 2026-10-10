"""Compose NVIDIA's released 2B configuration; customize only our experiment."""
import copy,json,os
from pathlib import Path
from omegaconf import OmegaConf
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.config_helper import override
from cosmos_predict2._src.predict2.configs.video2world.config import make_config as official_config
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader,get_sampler
from cosmos_predict2._src.predict2.configs.common.defaults.callbacks import BASIC_CALLBACKS,WANDB_CALLBACK
from cosmos_predict2._src.imaginaire.functional.lr_scheduler import LambdaWarmUpCosineScheduler
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_optimizer
from cosmos_predict2._src.imaginaire.lazy_config import PLACEHOLDER
from projects.generic_acn75.data import CachedVideoDataset,CachedLatentInput
from projects.generic_acn75.model import MechanismModel,MechanismConfig
from projects.generic_acn75.text import LocalProcessor
from projects.generic_acn75.audit import TrainingAudit

SETTINGS=json.loads(Path(__file__).with_name('settings.json').read_text())
def make_config(online_encoding=False, settings_override=None):
    s=SETTINGS if settings_override is None else settings_override
    mode=os.environ.get('COSMOS_MECHANISM','A');phase=os.environ.get('COSMOS_PHASE','smoke')
    assert mode in s['modes'] and phase in ['smoke','train']
    c=override(official_config(),['--','experiment=predict2_lora_training_2b_cosmos_nemo_assets_txt'])
    c.defaults=[]
    c.job.project=s['wandb_project'];c.job.group=s['wandb_group'];c.job.name=f'{phase}_{mode}'
    if phase=='train' and s.get('run_name'):c.job.name=s['run_name']
    c.job.wandb_mode='disabled' if phase=='smoke' else 'online'
    c.upload_reproducible_setup=False
    OmegaConf.set_struct(c.model.config,False)
    mc=MechanismConfig(**dict(c.model.config))
    mc.mechanism=mode;mc.context_noise_scales=s['context_noise_scales'];mc.context_frames=s['context_latents']
    mc.context_noise_grid=s['context_noise_grid'];mc.checkpoint_model_storage=s['checkpoint_model_storage']
    assert s['lora_precision'] in ['float32','bfloat16']
    mc.preserve_fp32_lora=s['lora_precision']=='float32'
    mc.native_checkpoint_sha256=s['native_checkpoint_sha256']
    mc.native_checkpoint=s['native_checkpoint'];mc.text_weights=s['text_weights']
    mc.audit_directory=str(Path(c.job.path_local)/'audit')
    mc.precision=s['precision'];mc.fsdp_shard_size=1;mc.ema.enabled=False
    mc.state_t=(s['frames']-1)//4+1;mc.min_num_conditional_frames=mc.max_num_conditional_frames=s['context_latents']
    mc.conditional_frames_probs=None;mc.train_time_distribution=s['time_distribution'];mc.shift=s['shift']
    mc.lora_rank=s['lora_rank'];mc.lora_alpha=s['lora_alpha'];mc.lora_target_modules=s['lora_target_modules']
    mc.use_lora=True;mc.use_dora=False;mc.init_lora_weights=True
    mc.tokenizer.vae_pth=s['vae_checkpoint'];mc.tokenizer.chunk_duration=s['frames']
    mc.tokenizer.load_mean_std=False
    if not online_encoding:
        mc.input_data_key='latent'
        mc.tokenizer=L(CachedLatentInput)()
    mc.text_encoder_config.compute_online=False
    mc.text_encoder_config.model_config.tokenizer=L(LocalProcessor)(path=s['text_weights'])
    mc.text_encoder_config.model_config.model_config.name_or_path=s['text_weights']
    mc.net.atten_backend='torch'
    mc.conditioner.text.dropout_rate=0.0
    mc.conditioner.use_video_condition.dropout_rate=0.0
    mc.net.sac_config.mode='block_wise';mc.net.sac_config.every_n_blocks=1
    c.model=L(MechanismModel)(config=mc,_recursive_=False)
    dataset=L(CachedVideoDataset)(cache_root=s['cache_root'],limit=8 if phase=='smoke' else None)
    c.dataloader_train=L(get_generic_dataloader)(dataset=dataset,sampler=L(get_sampler)(dataset=dataset),batch_size=s['per_gpu_batch'],drop_last=True,num_workers=2,pin_memory=True)
    c.dataloader_val=copy.deepcopy(c.dataloader_train)
    c.trainer.seed=s['seed']
    c.trainer.cudnn.benchmark=s['cudnn_benchmark']
    c.trainer.distributed_parallelism='ddp';c.trainer.max_iter=1 if phase=='smoke' else s['max_steps']
    c.trainer.grad_accum_iter=s['grad_accum'];c.trainer.logging_iter=1;c.trainer.run_validation=False
    c.trainer.callbacks=copy.deepcopy({**BASIC_CALLBACKS,**WANDB_CALLBACK})
    c.trainer.callbacks.pop('wandb_10x',None)
    c.trainer.callbacks['training_audit']=L(TrainingAudit)(upload_checkpoints=s.get('upload_checkpoints',False))
    c.trainer.callbacks['grad_clip'].clip_norm=s['clip_norm']
    c.trainer.callbacks['compile_tokenizer'].enabled=False
    c.optimizer=L(get_base_optimizer)(model=PLACEHOLDER,lr=s['learning_rate'],weight_decay=s['weight_decay'],betas=s['betas'],eps=s['eps'],optim_type='adamw')
    c.scheduler=L(LambdaWarmUpCosineScheduler)(warm_up_steps=[s['warmup']],cycle_lengths=[s['max_steps']],f_start=[1/s['warmup']],f_max=[1.],f_min=[0.])
    c.checkpoint.type=None
    c.checkpoint.load_path='';c.checkpoint.load_training_state=False;c.checkpoint.strict_resume=False
    c.checkpoint.save_iter=1 if phase=='smoke' else s['save_every']
    c.checkpoint.save_to_object_store.enabled=False;c.checkpoint.load_from_object_store.enabled=False
    c.model_parallel.context_parallel_size=1
    return c
