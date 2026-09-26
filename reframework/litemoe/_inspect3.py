import gguf, os
P = '/home/samuel/Carnice-Qwen3.6-MoE-35B-A3B-APEX-I-Mini.gguf'
r = gguf.GGUFReader(P)

names = [t.name for t in r.tensors]
print('total tensors:', len(names))
for i in [0, 1, 2, 3]:
    print(f'--- blk.{i} tensors ---')
    for t in r.tensors:
        if t.name.startswith(f'blk.{i}.'):
            print(f'  {t.name:70s} shape={tuple(t.shape)} type={t.tensor_type.name} n={t.n_bytes}')
print('--- non-blk (embed/norm/final) ---')
for t in r.tensors:
    n = t.name
    if not n.startswith('blk.'):
        print(f'  {n:70s} shape={tuple(t.shape)} type={t.tensor_type.name} n={t.n_bytes}')
# unique sublayer suffixes
import collections
suf = collections.Counter()
for n in names:
    if n.startswith('blk.'):
        rest = n.split('.', 2)[2]
        suf['.'.join(rest.split('.')[:2])] += 1
print('--- sublayer kinds (blk.X.<part1.part2> counts) ---')
for k, v in suf.most_common(40):
    print(f'  {k:50s} x{v}')
print('tiny model dir:')
for f in sorted(os.listdir('/home/samuel/Re/.hf/tiny-qwen3-moe')):
    print('  ', f, os.path.getsize('/home/samuel/Re/.hf/tiny-qwen3-moe/'+f))
