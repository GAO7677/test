"""Optional online metrics; disabled unless explicitly enabled."""
def start_run(settings, *, name, directory, config, smoke=False):
 if smoke or not settings.get('enabled',False):return None
 import wandb
 run=wandb.init(project=settings.get('project','fewshot-generation'),entity=settings.get('entity'),name=name,mode='online',dir=str(directory),config=config,settings=wandb.Settings(disable_code=True,disable_git=True))
 if run.offline:raise RuntimeError('W&B was explicitly enabled but online logging is unavailable')
 return run
