"""Reproducible FICTIONAL underwriting/outcome illustrations. Not actuarial data.

Label=1 represents a SIMULATED adverse loss next coverage term. It does not
represent claim fraud, a real underwriting decision, or observed policy losses.
"""
from pathlib import Path
import csv
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def generate(out_dir=ROOT/'sample_data',seed=941):
    out_dir=Path(out_dir)
    rng=np.random.default_rng(seed)
    names=sorted(p.name.removesuffix('_SYNTHETIC.csv') for p in out_dir.glob('*_SYNTHETIC.csv')
                 if not p.name.startswith('underwriting_'))
    for name in names:
        source=pd.read_csv(out_dir/f'{name}_SYNTHETIC.csv')
        destination=out_dir/f'underwriting_{name}_SYNTHETIC.csv'
        base_days=45 if name=='travel' else 210 if name=='crop' else 365
        durations=rng.integers(7,90,len(source)) if name=='travel' else (
            rng.integers(90,365,len(source)) if name=='crop' else rng.integers(180,730,len(source)))
        # Simulate prospective loss propensity from PRE-LOSS underwriting inputs.
        cover=source['coverage_amount'].to_numpy(float)
        prior=source['prior_claims'].to_numpy(float)
        relative_cover=cover / max(float(np.median(cover)),1.)
        z=-2.45 + 0.48*np.minimum(prior,8) + 0.46*np.log1p(relative_cover) + 0.32*durations/base_days
        propensity=1/(1+np.exp(-z))
        events=(rng.random(len(source))<propensity).astype(int)
        with destination.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['entity_group','coverage_amount','prior_claims',
                'policy_tenure_days','underwriting_loss_label'])
            writer.writeheader()
            for i in range(len(source)):
                writer.writerow({'entity_group':f'{name}_fictional_policy_{i}',
                                 'coverage_amount':round(float(cover[i]),2),
                                 'prior_claims':int(prior[i]),
                                 'policy_tenure_days':int(durations[i]),
                                 'underwriting_loss_label':int(events[i])})
    print(f'Generated {len(names)} FICTIONAL underwriting datasets. No actual insurers or actuarial observations.')

if __name__=='__main__':generate()
