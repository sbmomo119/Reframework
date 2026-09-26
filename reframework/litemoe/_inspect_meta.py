import gguf, json, struct
P = '/home/samuel/Carnice-Qwen3.6-MoE-35B-A3B-APEX-I-Mini.gguf'
r = gguf.GGUFReader(P)
print('=== ALL FIELDS ===')
for f in r.fields:
    t, name, payload, off, ln = f
    tv = int(t)
    if tv == 8:  # string
        v = payload.rstrip(b'\0').decode('utf-8', 'replace')
    elif tv == 4:
        v = struct.unpack('<i', payload)[0]
    elif tv == 5:
        v = struct.unpack('<q', payload)[0]
    elif tv == 6:
        v = struct.unpack('<f', payload)[0]
    elif tv == 7:
        v = struct.unpack('<d', payload)[0]
    elif tv == 9:  # array
        v = payload[:128].hex()
    else:
        v = repr(payload[:64])
    print(f'{name:60s} = {v}')
print('=== tensor name sample: router-ish ===')
for t in r.tensors:
    n = t.name
    if 'gate' in n and 'exps' not in n and 'shexp' not in n:
        print(f'  {n:70s} shape={tuple(int(s) for s in t.shape)} type={t.tensor_type.name}')
