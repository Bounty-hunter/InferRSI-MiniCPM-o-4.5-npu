"""CPU-only paired kernel-shape analysis of fixed-work NPU profiles."""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
root = args.root
all_groups = {}
provenance = {}
for role in ('A', 'B'):
    paths = list((root / f'profile-{role}').rglob('kernel_details.csv'))
    assert len(paths) == 1, (role, paths)
    path = paths[0]
    groups = defaultdict(lambda: dict(durations=[], blocks=Counter(), mix_blocks=Counter(), hf32=Counter(), counters=defaultdict(list)))
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        for row in reader:
            key = tuple(row.get(name, '') for name in ('Name','Type','Accelerator Core','Input Shapes','Input Data Types','Output Shapes'))
            duration = float(row['Duration(us)'])
            if duration <= 0:
                continue
            record = groups[key]
            record['durations'].append(duration)
            record['blocks'][row.get('Block Num', '')] += 1
            record['mix_blocks'][row.get('Mix Block Num', '')] += 1
            record['hf32'][row.get('HF32 Eligible', '')] += 1
            for name in ('aicore_time(us)','aic_mac_ratio','aic_scalar_ratio','aic_mte2_ratio',
                         'aiv_time(us)','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio'):
                value = row.get(name, '')
                if value and value != 'N/A':
                    record['counters'][name].append(float(value))
    all_groups[role] = groups
    provenance[role] = dict(path=str(path), bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), columns=columns,
                            rows=sum(len(group['durations']) for group in groups.values()), groups=len(groups))


def summary(group):
    values = group['durations']
    return dict(count=len(values), mean_us=statistics.mean(values), median_us=statistics.median(values),
                work_sum_ms=sum(values)/1000, block_num=dict(group['blocks']), mix_block_num=dict(group['mix_blocks']),
                hf32_eligible=dict(group['hf32']), counters_mean={name:statistics.mean(values) for name,values in group['counters'].items()})


paired = []
for key in all_groups['A'].keys() & all_groups['B'].keys():
    a, b = summary(all_groups['A'][key]), summary(all_groups['B'][key])
    paired.append(dict(name=key[0], op_type=key[1], core=key[2], input_shapes=key[3],
                       input_dtypes=key[4], output_shapes=key[5], A=a, B=b,
                       same_count=a['count']==b['count'],
                       mean_reduction_pct=(1-b['mean_us']/a['mean_us'])*100,
                       work_difference_ms=a['work_sum_ms']-b['work_sum_ms']))
paired.sort(key=lambda row:row['work_difference_ms'], reverse=True)
same_count = [row for row in paired if row['same_count']]
matched_work = {role:sum(row[role]['work_sum_ms'] for row in same_count) for role in ('A','B')}
all_work = {role:sum(sum(group['durations'])/1000 for group in groups.values()) for role,groups in all_groups.items()}
result = dict(scope='Paired fixed-work diagnostic profiles, 3 serial and3 multistream engine4 windows. Kernel work sums may overlap; not native latency.',
              provenance=provenance, paired_groups=len(paired), same_count_groups=len(same_count),
              matched_same_count_work_ms=matched_work, all_kernel_work_ms=all_work,
              largest_work_reductions=same_count[:30], largest_work_regressions=sorted(same_count,key=lambda row:row['work_difference_ms'])[:15],
              largest_control_work=sorted(paired,key=lambda row:row['A']['work_sum_ms'],reverse=True)[:30],
              unmatched={role:[dict(name=key[0],input_shapes=key[3],**summary(group)) for key,group in groups.items()
                               if key not in all_groups['B' if role=='A' else 'A']] for role,groups in all_groups.items()})
(root / 'profile-kernel-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(paired_groups=len(paired), same_count_groups=len(same_count),
                     matched_work_ms=matched_work, all_work_ms=all_work,
                     top=[dict(name=row['name'],shape=row['input_shapes'],
                               A=row['A']['mean_us'],B=row['B']['mean_us'],
                               blocks_A=row['A']['block_num'],blocks_B=row['B']['block_num'],
                               reduction_pct=row['mean_reduction_pct']) for row in same_count[:8]]),indent=2))
