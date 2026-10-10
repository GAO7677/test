import importlib.util,sys,json,os
from pathlib import Path
repo=Path('/home/gaoya/Code_Video/Code_MODEL/cosmos-predict2.5')
sys.path.insert(0,str(repo));os.chdir(repo)
root=Path('/data/gaoya/agent-data/outputs/inference_reproduction_20261010')
previous=root/'predict25_clean';out=root/'predict25_original_control';out.mkdir(parents=True,exist_ok=False)
p=json.loads((previous/'reproduction_plan.json').read_text())
from_source=Path('/data/gaoya/AAA_test_video/Context_Video_bench/Shared/supplement/gap038_triplet/predict25/N75/step1500/clean/hooks_snapshot.py')
source=from_source.read_text().replace('/data/gaoya/agent-data/outputs_v1/geometry_sweep_bench/20261007_context_triplet_gap038/predict_shared_context.pt','/data/gaoya/AAA_test_video/Context_Video_bench/Shared/supplement/gap038_triplet/predict_shared_context.pt')
namespace={};exec(compile(source,str(from_source),'exec'),namespace)
namespace['install_predict']('clean','N75',20261007,str(out/'triplet_audit.json'))
from projects.generic_acn75.infer_bench import main
sys.argv=['infer_bench.py','--settings',str(previous/'settings.json'),'--benchmark','test70','--method','N75','--step','1500','--adapter',p['adapter'],'--training-settings',str(previous/'training_settings.json'),'--case','v2v_gap_038','--output',str(out)]
main()
