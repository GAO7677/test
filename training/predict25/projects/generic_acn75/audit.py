"""Validation callback registered with NVIDIA's unchanged trainer."""
from pathlib import Path
import json
import torch
from cosmos_predict2._src.imaginaire.utils.callback import Callback

class TrainingAudit(Callback):
    def __init__(self, upload_checkpoints=False):
        super().__init__()
        self.upload_checkpoints=upload_checkpoints

    def on_train_start(self, model, iteration=0):
        from projects.generic_acn75.config import SETTINGS
        self.cache_fingerprint=json.loads((Path(SETTINGS['cache_root'])/'contract.json').read_text())['fingerprint']
        self.frozen = {n:(p._version,p.detach().flatten()[:8].clone()) for n,p in model.net.named_parameters() if not p.requires_grad}
        self.trainable = {n:p.detach().flatten()[:32].clone() for n,p in model.net.named_parameters() if p.requires_grad}
        self.grad_finite=False
        self.rank=torch.distributed.get_rank()
        assert self.trainable and all('lora_' in n for n in self.trainable)
        path=Path(model.config.audit_directory);path.mkdir(parents=True,exist_ok=True)
        (path/f'upload_policy_rank{self.rank}.json').write_text(json.dumps(dict(
            upload_checkpoints=self.upload_checkpoints,
            local_save_every=self.config.checkpoint.save_iter,
            wandb_mode=self.config.job.wandb_mode),indent=2))

    def on_before_optimizer_step(self, model_ddp, optimizer, scheduler, grad_scaler, iteration=0):
        if iteration != 0:return
        model=getattr(model_ddp,'module',model_ddp)
        grads=[p.grad for p in model.net.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        self.grad_finite=True

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        if iteration != 1:return
        params=dict(model.net.named_parameters())
        for n,(version,probe) in self.frozen.items():
            assert params[n]._version==version and torch.equal(probe,params[n].detach().flatten()[:8]),n
        changed=any(not torch.equal(probe,params[n].detach().flatten()[:32]) for n,probe in self.trainable.items())
        assert changed and self.grad_finite
        report=dict(status='PASS',optimizer_steps=iteration,finite_gradients=True,
                    only_lora_trainable=True,lora_changed=True,frozen_parameters_unchanged=True,
                    loss=float(loss.detach()),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    per_gpu_batch=data_batch[model.input_data_key].shape[0],grad_accum=self.config.trainer.grad_accum_iter,
                    vae_loaded=hasattr(model.tokenizer,'model'),text_encoder_loaded=model.text_encoder is not None,
                    cache_fingerprint=self.cache_fingerprint,
                    world_size=torch.distributed.get_world_size())
        path=Path(model.config.audit_directory);path.mkdir(parents=True,exist_ok=True)
        (path/f'validation_rank{self.rank}.json').write_text(json.dumps(report,indent=2))
        self.frozen.clear();self.trainable.clear()

    def on_save_checkpoint(self, model, state_dict):
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file,load_file
        import wandb
        iteration=state_dict['iteration']
        # Official checkpointer still saves optimizer/scheduler/scaler state.
        # The frozen base is identified separately, so keep only adapter weights
        # and training counters in the model section.
        assert model.config.checkpoint_model_storage in ['lora_and_training_counters','full']
        if model.config.checkpoint_model_storage=='lora_and_training_counters':
            state_dict['model']={k:v for k,v in state_dict['model'].items() if 'lora_' in k or '.accum_' in k}
        state_dict['base_checkpoint']=model.config.native_checkpoint
        folder=Path(self.config.job.path_local)/'adapters'/f'step{iteration:06d}'
        folder.mkdir(parents=True,exist_ok=True)
        adapter={k:v.detach().cpu().contiguous() for k,v in get_peft_model_state_dict(model.net).items()}
        save_file(adapter,str(folder/'adapter.safetensors'))
        restored=load_file(str(folder/'adapter.safetensors'))
        assert adapter.keys()==restored.keys() and all(torch.equal(v,restored[k]) for k,v in adapter.items())
        metadata=dict(iteration=iteration,mode=model.config.mechanism,
                      base_checkpoint=model.config.native_checkpoint,framework='nvidia-native',
                      adapter_roundtrip_exact=True)
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
        if self.upload_checkpoints and self.config.job.name.startswith('train_'):
            assert wandb.run is not None
            artifact=wandb.Artifact(f'predict25-native-{model.config.mechanism}',type='model',metadata=metadata)
            artifact.add_dir(str(folder))
            wandb.log_artifact(artifact,aliases=[f'step{iteration}','latest'])
