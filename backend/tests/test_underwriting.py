"""Regression coverage for independent underwriting training and risk analysis."""
from pathlib import Path
from app.training.underwriting import (train_underwriting,predict_underwriting,
                                       features_from_policy, inspect_underwriting_csv)
from app.training.tabular import FEATURES as FRAUD_FEATURES
ROOT=Path(__file__).resolve().parents[2]


def test_independent_underwriting_training_and_abstention(tmp_path):
    source=ROOT/'sample_data/underwriting_health_SYNTHETIC.csv'
    report=inspect_underwriting_csv(source)
    assert report['rows']>=100 and not report['issues']
    artifact=tmp_path/'underwriting.joblib'
    metrics=train_underwriting(source,artifact)
    assert artifact.exists() and metrics['held_out_test']['n']>0
    good={'policy_start':'2026-01-01','policy_end':'2026-12-31',
          'coverage_amount':300000,'prior_claims':0}
    assert len(features_from_policy(good))==3
    pred=predict_underwriting(good,artifact)
    assert pred['status']=='demo_inference',pred
    assert 0<=pred['demo_index']<=100 and pred['calibrated'] is False
    assert pred['outcome_definition'].startswith('SIMULATED')
    assert not (set(FRAUD_FEATURES)-{'coverage_amount','prior_claims'}) & set(pred['features'])
    missing=predict_underwriting({'policy_start':'2026-01-01'},artifact)
    assert missing['status']=='insufficient_data' and 'demo_index' not in missing
    ood=predict_underwriting({**good,'coverage_amount':1e11},artifact)
    assert ood['status']=='out_of_distribution' and 'demo_index' not in ood


def test_upload_train_deploy_underwriting_via_admin(client,auth,analyst):
    source=ROOT/'sample_data/underwriting_health_SYNTHETIC.csv'
    body={'category':'health','task':'underwriting_risk','source':'Synthetic test simulator',
          'label_definition':'1=simulated next-term loss event',
          'permitted_use':'Software tests', 'synthetic':'true'}
    unauth=client.post('/api/admin/datasets',headers=analyst,
                       data=body,files={'file':('uw.csv',source.read_bytes(),'text/csv')})
    assert unauth.status_code==403
    uploaded=client.post('/api/admin/datasets',headers=auth,
                         data=body,files={'file':('uw.csv',source.read_bytes(),'text/csv')})
    assert uploaded.status_code==201,uploaded.text
    dataset=uploaded.json();assert dataset['task']=='underwriting_risk'
    trained=client.post('/api/admin/train/'+dataset['id'],headers=auth)
    assert trained.status_code==200,trained.text
    assert trained.json()['task']=='underwriting_risk'
    deployed=client.post('/api/admin/models/'+trained.json()['model_id']+'/deploy',headers=auth)
    assert deployed.status_code==200,deployed.text
    case=client.post('/api/cases',headers=auth,json={'category':'health','kind':'application',
        'data':{'policy_number':'NEW-APP-001','coverage_amount':300000,'prior_claims':0,
                'policy_start':'2026-01-01','policy_end':'2026-12-31'}})
    assert case.status_code==201,case.text
    analysis=client.post('/api/cases/'+case.json()['id']+'/analyze',headers=auth)
    assert analysis.status_code==200,analysis.text
    result=analysis.json()['result']
    assert result['underwriting_model']['status']=='demo_inference'
    assert result['underwriting_risk_score'] is not None
    assert result['fraud_suspicion_score'] is None
    assert result['underwriting_model']['training_data_kind']=='synthetic_demo'
    assert result['underwriting_model_input_completeness_pct']==100
    assert result['underwriting_status']=='demo_inference'
    assert result['human_review_required'] is True
    reg=client.get('/api/insurance-types',headers=auth).json()
    assert reg['health']['underwriting_model_readiness']=='synthetic_demo_only'


def test_provision_two_tasks_idempotently_preserves_existing(client,auth):
    r=client.post('/api/admin/demo-models',headers=auth,json={'categories':['crop','commercial']})
    assert r.status_code==200,r.text
    items=r.json()['results']
    assert len(items)==4
    assert {x['task'] for x in items}=={'fraud_suspicion','underwriting_risk'}
    assert len({x['model_id'] for x in items})==4
    again=client.post('/api/admin/demo-models',headers=auth,json={'categories':['crop','commercial']})
    assert again.status_code==200
    assert all(x['status']=='already_active' for x in again.json()['results'])
    models=client.get('/api/models',headers=auth).json()
    assert len([m for m in models if m['deployed']])==4
    assert {m['task'] for m in models}=={'fraud_suspicion','underwriting_risk'}
    bad=client.post('/api/admin/datasets',headers=auth,data={'category':'crop','task':'bad',
        'source':'x','label_definition':'x','permitted_use':'x'},files={'file':('a.csv',b'test')})
    assert bad.status_code==422
