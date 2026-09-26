import gguf, json, os

P = '/home/samuel/Carnice-Qwen3.6-MoE-35B-A3B-APEX-I-Mini.gguf'
r = gguf.GGUFReader(P)

def val(f):
    # f = (type, name, payload, off, len) — decode payload heuristically
    t = f[0]
    if t == 4:   # string
        return bytes(f[2][1:]).decode('utf-8', 'replace')
    return f[2]

print('=== ARCH KV ===')
for k, f in sorted(r.fields.items()):
    if k.startswith('qwen35moe.') or k.startswith('general.'):
        try:
            print(f'  {k:50s} = {val(f)}')
        except Exception as e:
            print(f'  {k} -> err {e}')

print('\n=== ALL blk.0 TENSORS ===')
for t in r.tensors:
    if t.name.startswith('blk.0.'):
        print(f'  {t.name:42s} shape={list(t.shape)} type={t.tensor_type} bytes={t.n_bytes}')
print('  ... non-blk:')
for t in r.tensors:
    if not t.name.startswith('blk.'):
        print(f'  {t.name:42s} shape={list(t.shape)} type={t.tensor_type} bytes={t.n_bytes}')

print('\n=== TINY MODEL ===')
base = '/home/samuel/Re/.hf/tiny-qwen3-moe'
for root, dirs, files in os.walk(base):
    for f in files:
        p = os.path.join(root, f)
        print(f'  {p}  {os.path.getsize(p)} bytes')
cfgp = os.path.join(base, 'config.json')
if os.path.exists(cfgp):
    print('--- config.json ---')
    print(open(cfgp).read()[:2500])
