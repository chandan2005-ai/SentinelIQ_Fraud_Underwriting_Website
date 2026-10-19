"""Opt-in, idempotent deployment of independent fraud and underwriting DEMO models.

Never trains on uploaded claims, never replaces existing deployed models, and
never represents fictional outcomes as verified historical insurance outcomes.
"""
from uuid import uuid4
from sqlalchemy import select
from app.core.config import ROOT,DATA_DIR
from app.insurance_types.registry import SPECIFIC
from app.database.models import Dataset,ModelVersion
from app.training.tabular import train_tabular,inspect_csv,FEATURES
from app.training.underwriting import train_underwriting,inspect_underwriting_csv,FEATURES as UW_FEATURES

TASKS={
    'fraud_suspicion':{'prefix':'','inspect':inspect_csv,'train':train_tabular,'fields':FEATURES,
                      'label':'SIMULATED fraud_label'},
    'underwriting_risk':{'prefix':'underwriting_','inspect':inspect_underwriting_csv,
                         'train':train_underwriting,'fields':UW_FEATURES,'label':'SIMULATED underwriting_loss_label'}
}

def provision_synthetic(db,categories,actor_id=None,tasks=None):
    if not categories or any(cat not in SPECIFIC for cat in categories):
        raise ValueError('Select one or more supported insurance categories')
    tasks=tasks or list(TASKS)
    if not tasks or any(task not in TASKS for task in tasks):
        raise ValueError('Unknown or empty model tasks')
    output=[]
    for category in dict.fromkeys(categories):
        for task in dict.fromkeys(tasks):
            config=TASKS[task]
            existing=db.scalar(select(ModelVersion).where(ModelVersion.category==category,
                ModelVersion.task==task,ModelVersion.deployed.is_(True)))
            if existing:
                output.append({'category':category,'task':task,'status':'already_active',
                               'model_id':existing.id,'synthetic':existing.is_synthetic})
                continue
            source=ROOT/'sample_data'/f'{config["prefix"]}{category}_SYNTHETIC.csv'
            if not source.is_file():
                raise ValueError(f'Missing {task} synthetic training data: {source.name}')
            info=config['inspect'](source)
            if info['issues']:raise ValueError('; '.join(info['issues']))
            info['training_task']=task
            ds_id,model_id=str(uuid4()),str(uuid4())
            ds_name=ds_id+'.csv';model_name=model_id+'.joblib'
            (DATA_DIR/'datasets'/ds_name).write_bytes(source.read_bytes())
            try:
                metrics=config['train'](source,DATA_DIR/'model_artifacts'/model_name)
                metrics['data_is_synthetic']=True
                ds=Dataset(id=ds_id,category=category,file_name=ds_name,
                    source='Bundled reproducible fictional simulation; never a real-world policy history',
                    label_definition=config['label']+'; NOT real-world verified outcomes',
                    permitted_use='Research / local datathon demo ONLY',is_synthetic=True,quality=info)
                db.add(ds)
                db.add(ModelVersion(id=model_id,category=category,task=task,
                    artifact_name=model_name,dataset_id=ds_id,
                    feature_contract={'fields':config['fields'],'label':config['label']},metrics=metrics,
                    approval_status='demo_approved',is_synthetic=True,deployed=True))
                db.commit()
            except Exception:
                db.rollback()
                (DATA_DIR/'datasets'/ds_name).unlink(missing_ok=True)
                (DATA_DIR/'model_artifacts'/model_name).unlink(missing_ok=True)
                raise
            output.append({'category':category,'task':task,'status':'trained_and_deployed',
                           'model_id':model_id,'synthetic':True,'metrics':metrics['held_out_test']})
    return output
