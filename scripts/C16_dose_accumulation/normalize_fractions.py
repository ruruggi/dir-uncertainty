from dirunc_paths import LUNG_SBRT  # data roots, see README
import json, os
path = rf'{LUNG_SBRT}/Results_Accumulated_1_3/fractions.json'
with open(path, 'r') as f:
    spec = json.load(f)
for fx in spec['fractions']:
    fx['dvf'] = fx['dvf'].replace('\\', '/')
    fx['moving_dose'] = fx['moving_dose'].replace('\\', '/')
with open(path, 'w') as f:
    json.dump(spec, f, indent=2)
print('normalized')
print(json.dumps(spec, indent=2))
