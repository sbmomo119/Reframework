"""Diag: env, deps, RAM/disk, network. Results to /tmp/diag.txt."""
import subprocess, os, sys

out = []
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    out.append(f'>>> {cmd}\n{r.stdout[-2000:]}\n{r.stderr[-1000:]}')

sh('free -g | head -3')
sh('nproc')
sh('df -h /home/samuel | tail -1')
sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -3')
sh('cat /proc/cpuinfo | grep "model name" | head -1')

out.append('\n=== .venv-int4 python ===')
sh('.venv-int4/bin/python -V')
sh('.venv-int4/bin/python -c "import numpy; print(\'numpy\', numpy.__version__)" 2>&1')
sh('.venv-int4/bin/python -c "import torch; print(\'torch\', torch.__version__, torch.version.cuda, torch.cuda.is_available())" 2>&1')
sh('.venv-int4/bin/python -c "import gguf; print(\'gguf\', gguf.__version__ if hasattr(gguf,"__version__") else "ok")" 2>&1')
sh('.venv-int4/bin/python -c "import safetensors; print(\'safetensors ok\')" 2>&1')
sh('.venv-int4/bin/pip list 2>/dev/null | head -30')

out.append('\n=== network ===')
sh('curl -sI --max-time 6 https://raw.githubusercontent.com 2>&1 | head -2; echo rc=$?')
sh('curl -sI --max-time 6 https://huggingface.co 2>&1 | head -2; echo rc=$?')

out.append('\n=== litemoe dir so far ===')
sh('ls -la reframework/litemoe/ 2>/dev/null')
sh('ls reframework/ | head -30')

open('/tmp/diag.txt','w').write('\n'.join(out))
print('done')
