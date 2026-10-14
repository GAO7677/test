"""Scheduling boundaries without loading a CUDA model or changing the database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.fewshot_service import gpu_admission

total = 49140
shared = {'allow_shared_gpu': True, 'gpu_min_free_mb': 24576, 'gpu_reserve_mb': 2048}
assert gpu_admission({}, (512, total-512, total)) == (True, 0.9)
assert gpu_admission({}, (20000, total-20000, total)) == (False, 0.9)
allowed, fraction = gpu_admission(shared, (20000, total-20000, total))
assert allowed and abs(fraction * total - (total-20000-2048)) < 1e-6
assert not gpu_admission(shared, (total-24575, 24575, total))[0]
assert gpu_admission(shared, (total-24576, 24576, total))[0]
assert not gpu_admission(shared, (35000, total-35000, total))[0]
try:
    gpu_admission(dict(shared, gpu_reserve_mb=24576), (0, total, total))
except ValueError:
    pass
else:
    raise AssertionError('Accepted invalid GPU memory policy')
print('PASS: busy GPU sharing, low-memory queue, allocation cap, threshold boundary, legacy idle policy')
