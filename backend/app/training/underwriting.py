"""Separate RESEARCH underwriting model: simulated next-term adverse loss event.

Only ex-ante fields are used. NO claim amount, invoice, suspected fraud,
medical diagnosis, sensitive personal attributes or claim adjudication data.
Numerical result is a fictional-data decision-support INDEX, not a quoted
premium, actual probability of loss, or approval/decline recommendation.
"""
from datetime import date,datetime,timezone
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,brier_score_loss

FEATURES=['coverage_amount','prior_claims','policy_tenure_days']
LABEL='underwriting_loss_label'


def features_from_policy(data):
    def number(key):
        try:
            value=float(data.get(key))
            return value if math.isfinite(value) else float('nan')
        except (ValueError,TypeError):return float('nan')
    try:
        a=date.fromisoformat(str(data['policy_start']))
        b=date.fromisoformat(str(data['policy_end']))
        duration=float((b-a).days+1) if b>=a else float('nan')
    except (ValueError,TypeError,KeyError): duration=float('nan')
    return {'coverage_amount':number('coverage_amount'), 'prior_claims':number('prior_claims'),
            'policy_tenure_days':duration}


def inspect_underwriting_csv(path):
    df=pd.read_csv(path)
    issues=[f'Missing feature {key}' for key in FEATURES if key not in df.columns]
    if LABEL not in df: issues.append(f'Missing outcome {LABEL}')
    elif set(df[LABEL].dropna().unique())-{0,1}: issues.append('Underwriting outcome must be binary 0/1')
    return {'rows':len(df),'columns':list(df.columns),'duplicate_rows':int(df.duplicated().sum()),
            'missing_pct':{k:round(float(v)*100,2) for k,v in df.isna().mean().items()},
            'label_counts':{str(k):int(v) for k,v in df[LABEL].value_counts(dropna=False).items()} if LABEL in df else {},
            'training_task':'underwriting_risk','issues':issues}


def train_underwriting(path,artifact,seed=43):
    quality=inspect_underwriting_csv(path)
    if quality['issues']: raise ValueError('; '.join(quality['issues']))
    df=pd.read_csv(path).drop_duplicates().copy()
    for key in FEATURES+ [LABEL]:df[key]=pd.to_numeric(df[key],errors='coerce')
    df=df.dropna(subset=FEATURES+[LABEL])
    df=df[df[LABEL].isin([0,1])]
    if len(df)<100 or len(df[LABEL].unique())!=2 or df[LABEL].value_counts().min()<25:
        raise ValueError('Underwriting model requires >=100 complete labeled rows and >=25 per simulated outcome')
    if (df[FEATURES]<0).any().any() or (df['policy_tenure_days']==0).any() or (df['coverage_amount']==0).any():
        raise ValueError('Invalid coverage, prior-claim count or policy term')
    trainval,test=train_test_split(df,test_size=.2,random_state=seed,stratify=df[LABEL]);
    train,val=train_test_split(trainval,test_size=.25,random_state=seed,stratify=trainval[LABEL]);
    candidates={
        'logistic_regression':Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),
               ('model',LogisticRegression(max_iter=2000,random_state=seed))]),
        'random_forest':RandomForestClassifier(n_estimators=80,min_samples_leaf=6,max_depth=5,random_state=seed,n_jobs=1)
    }
    metrics={}
    for name,model in candidates.items():
        model.fit(train[FEATURES],train[LABEL].astype(int))
        p=model.predict_proba(val[FEATURES])[:,1]
        metrics[name]={'pr_auc':float(average_precision_score(val[LABEL],p)),
                       'roc_auc':float(roc_auc_score(val[LABEL],p)),
                       'brier':float(brier_score_loss(val[LABEL],p))}
    selected_name=max(metrics,key=lambda name:(metrics[name]['pr_auc'],-metrics[name]['brier']))
    chosen=candidates[selected_name]
    p=chosen.predict_proba(test[FEATURES])[:,1]
    held_out={'n':int(len(test)),'pr_auc':float(average_precision_score(test[LABEL],p)),
              'roc_auc':float(roc_auc_score(test[LABEL],p)),'brier':float(brier_score_loss(test[LABEL],p)),
              'precision_at_50pct':float(precision_score(test[LABEL],p>=.5,zero_division=0)),
              'recall_at_50pct':float(recall_score(test[LABEL],p>=.5,zero_division=0))}
    bounds={x:{'min':float(train[x].min()),'max':float(train[x].max())} for x in FEATURES}
    artifact=Path(artifact);artifact.parent.mkdir(exist_ok=True,parents=True)
    joblib.dump({'model':chosen,'task':'underwriting_risk','feature_names':FEATURES,'bounds':bounds,
                 'algorithm':selected_name,'version':datetime.now(timezone.utc).isoformat(),
                 'outcome_definition':'SIMULATED next-term adverse loss event; NOT observed insurance outcomes',
                 'calibrated_for_real_world':False},artifact)
    return {'algorithm':selected_name,'splits':{'train':len(train),'validation':len(val),'test':len(test)},
            'validation_comparison':metrics,'held_out_test':held_out,'data_is_synthetic':None,
            'label':'underwriting_loss_label','target_definition':'Simulated next-term adverse loss event',
            'calibrated':False,'warning':'Held-out metrics evaluate a FICTIONAL simulator only; no actuarial validation.'}


def predict_underwriting(data,artifact):
    # Only trusted joblib models created by this project may be loaded.
    pack=joblib.load(artifact)
    if pack.get('task')!='underwriting_risk': raise ValueError('Wrong task artifact for underwriting inference')
    features=features_from_policy(data)
    if not all(math.isfinite(v) for v in features.values()):
        return {'status':'insufficient_data','reason':'Provide policy start, policy end, coverage amount and prior claims',
                'missing_inputs':[key for key,value in features.items() if not math.isfinite(value)]}
    if features['coverage_amount']<=0 or features['prior_claims']<0 or features['policy_tenure_days']<=0:
        return {'status':'invalid_data','reason':'Coverage and policy duration must be positive; prior claims nonnegative'}
    out_of_range=[]
    for name,value in features.items():
        bound=pack['bounds'][name]; margin=max((bound['max']-bound['min'])*.25,1.)
        if value<bound['min']-margin or value>bound['max']+margin: out_of_range.append(name)
    if out_of_range:
        return {'status':'out_of_distribution','reason':'Policy values beyond simulated training domain',
                'out_of_range_features':out_of_range}
    row=pd.DataFrame([features],columns=FEATURES)
    value=float(pack['model'].predict_proba(row)[0,1])
    return {'status':'demo_inference','demo_index':round(value*100,2),
            'algorithm':pack['algorithm'],'model_version':pack['version'],
            'features':features,'outcome_definition':pack['outcome_definition'],
            'calibrated':False,'calibrated_for_real_world':False,
            'note':'Research simulation only; neither a real-world loss probability nor an underwriting verdict.'}
