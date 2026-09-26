"""Inspect the 14GB GGUF: quant types, expert tensor layout, per-layer arch."""
import gguf, re, collections

P = '/home/samuel/Carnice-Qwen3.6-MoE-35B-A3B-APEX-I-Mini.gguf'
r = gguf.GGUFReader(P)

t0 = r.tensors[0]
print('sample tensor:', t0.name, t0.shape, t0.tensor_type, t0.n_bytes, type(t0.data))

qts = collections.Counter()
for t in r.tensors:
    q = gguf.GGMLQuantizationType(int(t.tensor_type))
    qts[q.name] += 1
print('QUANT TYPES:', dict(qts))

names = [t.name for t in r.tensors]
pat = collections.Counter()
for n in names:
    p = re.sub(r'\d+', 'N', n)
    pat[p] += 1
print('\nNAME PATTERNS:')
for p, c in pat.most_common(60):
    print(f'  {c:5d}  {p}')

print('\nEXPERT TENSOR SHAPE (blk.0):')
for t in r.tensors:
    if t.name.startswith('blk.0.ffn_') and ('exps' in t.name or 'shexp' in t.name):
        print(f'  {t.name:40s} dims={len(t.shape)} shape={list(t.shape)} type={gguf.GGMLQuantizationType(int(t.tensor_type)).name} bytes={t.n_bytes}')

print('\nLAYER ARCH:')
blk_attn_qkv = [n for n in names if re.match(r'blk\.(\d+)\.attn_qkv', n)]
blk_ssm_a = [n for n in names if re.match(r'blk\.(\d+)\.ssm_a$', n)]
print('layers with attn_qkv (linear-attn):', len(blk_attn_qkv))
print('layers with ssm_a (mamba):', len(blk_ssm_a))
full = [n for n in names if re.match(r'blk\.(\d+)\.attn_q\.', n)]
print('layers with attn_q (full-attn):', len(full))

exp_bytes = sum(t.n_bytes for t in r.tensors if '.ffn_' in t.name and 'exps' in t.name and 'shexp' not in t.name)
sh_bytes = sum(t.n_bytes for t in r.tensors if 'shexp' in t.name)
print('\nexperts bytes: %.2f GiB' % (exp_bytes / 2**30))
print('shared-expert bytes: %.2f GiB' % (sh_bytes / 2**30))
print('file total: %.2f GiB' % (sum(t.n_bytes for t in r.tensors) / 2**30))

print('\nROUTING-LIKE TENSORS:')
for t in r.tensors:
    n = t.name
    if re.search(r'(gate|router)', n) and 'ffn_gate_inp' not in n:
        print(f'  {n:45s} dims={len(t.shape)} shape={list(t.shape)}')
