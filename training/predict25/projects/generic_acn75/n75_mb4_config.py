"""Single N75 production experiment; all run settings live in its JSON file."""
import json,os,time
from pathlib import Path
import torch
import wandb
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.callback import Callback
from projects.generic_acn75.config import make_config as base_config

RUN=json.loads(Path(__file__).with_name('n75_mb4_settings.json').read_text())

class RunRecord(Callback):
    def write(self,status,iteration,**extra):
        if torch.distributed.get_rank()!=0:return
        path=Path(RUN['output'])/'launch_status.json'
        record=dict(status=status,production_training_started=True,mode='N75',per_gpu_batch=4,
                    grad_accum=1,effective_batch=8,iteration=iteration,max_steps=RUN['max_steps'],
                    upload_checkpoints=RUN['upload_checkpoints'],cache_validation_policy=RUN['cache_validation_policy'],cache_reference_encoding_batch=2,
                    wandb_url=wandb.run.url,wandb_id=wandb.run.id,tmux='predict25_N75_mb4_official',
                    output=str(self.config.job.path_local),time=time.time(),**extra)
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)

    def on_train_start(self,model,iteration=0):
        if torch.distributed.get_rank()!=0:return
        assert wandb.run is not None and wandb.run.settings.mode=='online'
        wandb.run.config.update({'experiment_settings':RUN},allow_val_change=True)
        self.write('RUNNING',iteration)

    def on_training_step_end(self,model,data_batch,output_batch,loss,iteration=0):
        if torch.distributed.get_rank()==0:self.write('RUNNING',iteration,rank0_loss=float(loss.detach()))

    def on_train_end(self,model,iteration=0):self.write('COMPLETE',iteration)

def make_config():
    assert os.environ['COSMOS_MECHANISM']=='N75' and os.environ['COSMOS_PHASE']=='train'
    assert RUN['per_gpu_batch']*RUN['world_size']*RUN['grad_accum']==RUN['effective_batch']
    contract=json.loads((Path(RUN['cache_root'])/'contract.json').read_text())
    assert RUN['cache_validation_policy']=='fixed_validated_cache'
    assert contract['contract']['cache_batch_size']==RUN['cache_reference_encoding_batch']==2
    ready=json.loads((Path(RUN['cache_root'])/'ready.json').read_text())
    assert ready['status']=='PASS' and ready['fingerprint']==contract['fingerprint']
    config=base_config(settings_override=RUN)
    config.trainer.callbacks['run_record']=L(RunRecord)()
    return config
