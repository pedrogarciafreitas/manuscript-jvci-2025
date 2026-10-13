"""Scoped, resumable review analyses; source files and frozen results are immutable."""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/analysis'
FEATURES = ROOT / 'data/features'
KEY = ['SIGNAL', 'REF']
DATASETS = ['APSIPA', 'ICIP2023', 'LS-PCQA', 'SJTU-PCQA', 'UnB_PC', 'WPC', 'WPC2']
FAMILY_DATASETS = ['APSIPA', 'ICIP2023', 'SJTU-PCQA', 'UnB_PC']
METHODS = ['PointPCA', 'PointPCA+', 'PointPCA+RS', 'PointPCA3', 'PointPCA++']
REPEATS = 10000
SEED = 42
B = None


def path(kind, name):
    return OUT / kind / name


def read(kind, name):
    return pd.read_csv(path(kind, name))


def write(kind, name, frame):
    destination = path(kind, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def save_json(kind, name, value):
    destination = path(kind, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(FEATURES.rglob('*.csv'))}


def check_environment():
    recorded = json.loads(path('manifests', 'core3_selection.json').read_text())['packages']
    installed = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lazypredict']}
    for name, value in installed.items():
        assert value == recorded[name.replace('-', '_')], (name, value, recorded)
    return installed


def complete(dataset):
    frame = B.load_final_complete_dataset(dataset)
    assert not frame.duplicated(KEY).any()
    assert len(frame) == B.FINAL_EXPECTED_POPULATIONS[dataset][0]
    assert frame.REF.nunique() == B.FINAL_EXPECTED_POPULATIONS[dataset][1]
    assert np.isfinite(frame.SCORE).all()
    return frame


def align(base, other):
    assert not other.duplicated(KEY).any()
    assert set(map(tuple, base[KEY].values)) == set(map(tuple, other[KEY].values))
    ordered = other.set_index(KEY).loc[pd.MultiIndex.from_frame(base[KEY])].reset_index()
    for column in ['SCORE', 'REF_NUM_POINTS', 'TEST_NUM_POINTS', 'CLASS']:
        assert base[column].fillna('unknown').astype(str).equals(ordered[column].fillna('unknown').astype(str)), column
    return ordered


def assignments(dataset, base):
    folds = read('folds', 'final_pointpcapp_groupkfold.csv')
    folds = folds[folds.dataset.eq(dataset)]
    assert not folds.duplicated(KEY).any()
    assert set(map(tuple, folds[KEY].values)) == set(map(tuple, base[KEY].values))
    folds = folds.set_index(KEY).loc[pd.MultiIndex.from_frame(base[KEY]), 'fold'].to_numpy()
    assert len(np.unique(folds)) == 5
    assert pd.DataFrame({'ref': base.REF, 'fold': folds}).groupby('ref').fold.nunique().max() == 1
    return folds


def predictors(dataset, method, base):
    if method == 'PointPCA++':
        return base[B.final_feature_columns()].copy()
    if method == 'PointPCA3':
        return base[B.final_pc3_features()].copy()
    if method == 'DISTS':
        return base[B.final_dists_features()].copy()
    if method == 'PointPCA':
        file = FEATURES / 'pointpca/matlab/pointpca1' / dataset / 'PointPCA1.csv'
        names = [f'FV_PointPCA1_{i}' for i in range(46)]
    elif method == 'PointPCA+':
        file = FEATURES / 'pointpca/matlab/pointpca2/features' / dataset / 'PointPCA2_matlab.csv'
        names = [f'QUALITY_INDEX_PointPCA2_{i}' for i in range(40)]
    elif method == 'PointPCA+RS':
        file = FEATURES / 'pointpca/pointpca2-rs' / dataset / 'PointPCA2_rs.csv'
        names = [f'QUALITY_INDEX_PointPCA2_{i}' for i in range(40)]
    else:
        raise ValueError(method)
    raw = align(base, pd.read_csv(file))
    actual = [c for c in raw if c.startswith(('FV_', 'QUALITY_INDEX_'))]
    assert actual == names, (method, actual)
    assert np.isfinite(raw[names].to_numpy(dtype=float)).all()
    return raw[names].copy()


def frozen_signature():
    names = ['scripts/review_analysis.py', 'scripts/run_analysis.py', 'data/analysis/manifests/core3_selection.json', 'data/analysis/folds/final_pointpcapp_groupkfold.csv']
    return {name: sha(ROOT / name) for name in names}


def audit():
    versions = check_environment()
    provenance = json.loads(path('manifests', 'review_method_provenance.json').read_text())
    assert provenance['semantic_order_verified']
    dictionary = read('manifests', 'review_feature_dictionary.csv')
    assert dictionary.groupby('family').size().to_dict() == {'PointPCA': 46, 'PointPCA+': 40}
    sources = source_hashes()
    assert len(sources) == 300
    baseline_path = path('manifests', 'review_revision_manifest.json')
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        assert baseline['source_hashes'] == sources
        for name, value in baseline['frozen_artifact_hashes'].items():
            assert sha(ROOT / name) == value, name
    else:
        baseline = {'source_hashes': sources, 'frozen_artifact_hashes': {name: sha(ROOT / name) for name in subprocess.check_output(['git', 'ls-files', 'data/analysis'], cwd=ROOT, text=True).splitlines() if (ROOT / name).is_file()}, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    checks = []
    for dataset in DATASETS:
        base = complete(dataset)
        assignments(dataset, base)
        arms = ['PointPCA3', 'PointPCA++', 'DISTS']
        if dataset in FAMILY_DATASETS:
            arms += ['PointPCA+', 'PointPCA+RS']
        if dataset == 'APSIPA':
            arms += ['PointPCA']
        for method in arms:
            x = predictors(dataset, method, base)
            checks.append({'dataset': dataset, 'method': method, 'samples': len(x), 'references': base.REF.nunique(), 'predictors': x.shape[1], 'finite': bool(np.isfinite(x).all().all())})
    assert all(x['finite'] for x in checks)
    write('audits', 'review_inputs.csv', pd.DataFrame(checks))
    overlap = []
    for source in ['APSIPA', 'WPC', 'LS-PCQA']:
        for target in DATASETS:
            if target == source:
                continue
            refs = set(complete(source).REF) & set(complete(target).REF)
            for ref in sorted(refs):
                overlap.append({'train_dataset': source, 'test_dataset': target, 'REF': ref, 'identity_status': 'shared filename; raw-content identity unverified'})
    write('audits', 'review_transfer_overlap.csv', pd.DataFrame(overlap))
    registry = json.loads(path('manifests', 'run_manifest.json').read_text())['lazy_regressors']
    selected = json.loads(path('manifests', 'core3_selection.json').read_text())
    assert len(registry) == 41
    assert set(registry) == set(selected['regressor_details']['eligible_models'] + selected['regressor_details']['ineligible_models'])
    baseline.update({'packages': versions, 'frozen_model': 'ExtraTreesRegressor', 'features': 46, 'seed': SEED, 'resamples': REPEATS, 'comparison_datasets': FAMILY_DATASETS, 'historical_supplement_timings_used': False, 'selection_reopened': False, 'script_hashes': frozen_signature(), 'inference': 'reference-clustered conditional-on-fitted-predictions uncertainty; no formal equivalence', 'unavailable': {'resolution': 'raw APSIPA clouds and reproducible multi-resolution extraction/features', 'changed_configuration_cost': 'controlled timers that skip unused operations', 'hybrid_rs_cost': 'JVCI-compatible matched timings; historical predictor timings excluded', 'camera_geometry': 'historical index-to-camera mapping'}})
    save_json('manifests', 'review_revision_manifest.json', baseline)
    print('Audit passed: 300 source CSVs, complete shared-fold comparisons, 41 recorded models', flush=True)


def fit_arm(dataset, method, variant, base, x):
    fold = assignments(dataset, base)
    assert np.isfinite(x.to_numpy(dtype=float)).all()
    prediction = np.full(len(base), np.nan)
    for number in sorted(np.unique(fold)):
        train = fold != number
        test = ~train
        assert not set(base.loc[train, 'REF']) & set(base.loc[test, 'REF'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, prediction[test] = B.fit_named_model('ExtraTreesRegressor', x.loc[train], base.loc[train, 'SCORE'], x.loc[test])
    assert np.isfinite(prediction).all()
    result = base[KEY].copy()
    result.insert(0, 'dataset', dataset)
    result['method'] = method
    result['variant'] = variant
    result['fold'] = fold
    result['y_true'] = base.SCORE.to_numpy()
    result['y_pred'] = prediction
    result['features'] = x.shape[1]
    return result


def diagnostic_designs():
    base = complete('APSIPA')
    pc3 = base[B.final_pc3_features()].copy()
    blocks = {}
    for label, iqa in [('dists', 'DISTSMetric'), ('topiq', 'TOPIQ_FRMetric')]:
        frame = align(base, pd.read_csv(B.projection_path('crop_pad_navier', iqa)))
        blocks[label] = frame[[f'FV_Cubemap_{i}' for i in range(6)]].set_axis([f'{label}_{i}' for i in range(6)], axis=1)
    d, t = blocks['dists'], blocks['topiq']
    designs = {'pc3': ('iqa', pc3), 'dists': ('iqa', d), 'topiq': ('iqa', t), 'dists_topiq': ('iqa', pd.concat([d,t],axis=1)), 'pc3_dists': ('iqa',pd.concat([pc3,d],axis=1)), 'pc3_topiq': ('iqa',pd.concat([pc3,t],axis=1)), 'pc3_dists_topiq': ('iqa',pd.concat([pc3,d,t],axis=1))}
    for size in [2,4,6]:
        for indices in itertools.combinations(range(6), size):
            subset = d.iloc[:, list(indices)]
            for prefix, frame in [('views', subset), ('pc3_views', pd.concat([pc3,subset],axis=1))]:
                designs[prefix + '_' + ''.join(map(str,indices))] = ('views',frame)
    for operation in ['mean','median','min','max']:
        pooled = getattr(d, operation)(axis=1).to_frame('dists_' + operation)
        designs['pool_' + operation] = ('pooling',pooled)
        designs['pc3_pool_' + operation] = ('pooling',pd.concat([pc3,pooled],axis=1))
    return base, designs


def predictions():
    audit()
    frozen = read('predictions','final_pointpcapp_oof_predictions.csv')
    all_rows = []
    replay = []
    checkpoint = path('checkpoints','review_predictions')
    checkpoint.mkdir(exist_ok=True)
    signatures = frozen_signature()
    for dataset in DATASETS:
        base = complete(dataset)
        arms = ['PointPCA3', 'DISTS', 'PointPCA++']
        if dataset in FAMILY_DATASETS:
            arms += ['PointPCA+', 'PointPCA+RS']
        if dataset == 'APSIPA':
            arms += ['PointPCA']
        for method in arms:
            filename = checkpoint / f'{dataset}_{METHODS.index(method) if method in METHODS else "DISTS"}.csv'
            meta = filename.with_suffix('.json')
            if filename.exists() and meta.exists() and json.loads(meta.read_text()) == signatures:
                result = pd.read_csv(filename)
            else:
                print('Fitting',dataset,method,flush=True)
                result = fit_arm(dataset,method,'standalone',base,predictors(dataset,method,base))
                result.to_csv(filename,index=False)
                meta.write_text(json.dumps(signatures,sort_keys=True))
            if method == 'PointPCA++':
                comparison = result.merge(frozen[frozen.dataset.eq(dataset)],on=['dataset',*KEY,'fold'],suffixes=('','_frozen'),validate='one_to_one')
                error = float(np.abs(comparison.y_pred-comparison.y_pred_frozen).max())
                replay.append({'dataset':dataset,'max_absolute_prediction_difference':error})
                assert error < 1e-12, (dataset,error)
                result['y_pred'] = comparison.y_pred_frozen.to_numpy()
            all_rows.append(result)
    write('predictions','review_baselines.csv',pd.concat(all_rows,ignore_index=True))
    write('audits','review_frozen_reproduction.csv',pd.DataFrame(replay))
    base, designs = diagnostic_designs()
    rows = []
    for variant,(experiment,x) in designs.items():
        filename = checkpoint / ('APSIPA_diag_' + variant + '.csv')
        meta = filename.with_suffix('.json')
        if filename.exists() and meta.exists() and json.loads(meta.read_text()) == signatures:
            result = pd.read_csv(filename)
        else:
            print('Fitting APSIPA diagnostic',variant,flush=True)
            result = fit_arm('APSIPA','diagnostic',variant,base,x)
            result.to_csv(filename,index=False)
            meta.write_text(json.dumps(signatures,sort_keys=True))
        result['experiment'] = experiment
        rows.append(result)
    write('predictions','review_ablations.csv',pd.concat(rows,ignore_index=True))
    print('Predictions complete',flush=True)


def run(base_module, stage):
    global B
    B = base_module
    stages = {'audit': audit, 'predictions': predictions, 'statistics': lambda: statistics(), 'reports': lambda: reports(), 'validate': lambda: validate()}
    stages['presentation'] = lambda: __import__('review_presentation').render(__import__(__name__))
    for name in ([n for n in stages if n != 'presentation'] if stage == 'all' else [stage]):
        print('Review stage:',name,flush=True)
        stages[name]()


def weighted_corr(x, y, weights):
    """Pearson correlation of samples repeated according to nonnegative weights."""
    total = weights.sum(axis=1)
    mx = (weights*x).sum(axis=1)/total
    my = (weights*y).sum(axis=1)/total
    dx = x-mx[:,None]
    dy = y-my[:,None]
    denominator = np.sqrt((weights*dx*dx).sum(axis=1)*(weights*dy*dy).sum(axis=1))
    return np.divide((weights*dx*dy).sum(axis=1),denominator,out=np.full(len(total),np.nan),where=denominator>0)


def weighted_ranks(values, weights):
    """Midranks for a cluster bootstrap, including ties and duplicated clusters."""
    _, inverse = np.unique(values, return_inverse=True)
    order = np.argsort(inverse,kind='stable')
    starts = np.r_[0,np.flatnonzero(np.diff(inverse[order]))+1]
    mass = np.add.reduceat(weights[:,order],starts,axis=1)
    return (mass.cumsum(axis=1)-mass/2)[:,inverse]


def metric_vectors(y, prediction, weights):
    if np.ndim(prediction) == 1:
        rpred = weighted_ranks(prediction,weights)
    else:
        # Swapped prediction vectors have one observation per row, no resampling.
        rpred = rankdata(prediction,axis=1)
    rtrue = weighted_ranks(y,weights)
    return np.column_stack([weighted_corr(rtrue,rpred,weights),weighted_corr(y,prediction,weights),np.sqrt((weights*(prediction-y)**2).sum(axis=1)/weights.sum(axis=1))])


def scalar_metrics(y,prediction):
    return metric_vectors(np.asarray(y,float),np.asarray(prediction,float),np.ones((1,len(y))))[0]


def bootstrap_metrics(y, predictions_by_arm, refs):
    _, code = np.unique(refs,return_inverse=True)
    groups = code.max()+1
    rng = np.random.default_rng(SEED)
    result = {arm: [] for arm in predictions_by_arm}
    for start in range(0,REPEATS,256):
        count = min(256,REPEATS-start)
        multiplicity = rng.multinomial(groups,np.full(groups,1/groups),size=count)
        weights = multiplicity[:,code]
        for arm,prediction in predictions_by_arm.items():
            result[arm].append(metric_vectors(y,prediction,weights))
    return {arm:np.concatenate(values) for arm,values in result.items()}


def swapped_tests(y,a,b,refs):
    _,code = np.unique(refs,return_inverse=True)
    rng = np.random.default_rng(SEED)
    observed = scalar_metrics(y,a)-scalar_metrics(y,b)
    exceed = np.zeros(3,dtype=int)
    valid = np.zeros(3,dtype=int)
    for start in range(0,REPEATS,256):
        n = min(256,REPEATS-start)
        swaps = rng.integers(0,2,size=(n,code.max()+1)).astype(bool)[:,code]
        aa = np.where(swaps,b,a)
        bb = np.where(swaps,a,b)
        weights = np.ones_like(aa)
        delta = metric_vectors(y,aa,weights)-metric_vectors(y,bb,weights)
        finite = np.isfinite(delta)
        exceed += ((np.abs(delta)>=np.abs(observed)-1e-14)&finite).sum(axis=0)
        valid += finite.sum(axis=0)
    return (exceed+1)/(valid+1),REPEATS-valid


def holm(pvalues):
    values=np.asarray(pvalues,float)
    assert np.isfinite(values).all()
    order=np.argsort(values)
    adjusted=np.minimum(1,np.maximum.accumulate((len(values)-np.arange(len(values)))*values[order]))
    result=np.empty(len(values));result[order]=adjusted
    return result


def summarize_predictions(frame, arm_column, paired=False):
    summaries=[];folds=[];tests=[]
    for dataset,population in frame.groupby('dataset',sort=False):
        arms={};boot={}
        template=None
        for arm,part in population.groupby(arm_column,sort=False):
            part=part.sort_values(KEY).reset_index(drop=True)
            if template is None:
                template=part
            assert np.array_equal(part[KEY].values,template[KEY].values)
            assert np.array_equal(part.y_true,template.y_true)
            assert np.array_equal(part.fold,template.fold)
            arms[arm]=part.y_pred.to_numpy()
            for number,f in part.groupby('fold'):
                values=scalar_metrics(f.y_true,f.y_pred)
                folds.append({'dataset':dataset,arm_column:arm,'fold':number,**dict(zip(['srocc','plcc','rmse'],values))})
        y=template.y_true.to_numpy();refs=template.REF.to_numpy()
        print('Bootstrap',dataset,len(arms),'arms',flush=True)
        boot=bootstrap_metrics(y,arms,refs)
        for arm,prediction in arms.items():
            values=scalar_metrics(y,prediction)
            row={'dataset':dataset,arm_column:arm,'samples':len(y),'references':len(np.unique(refs)),'features':int(population.loc[population[arm_column].eq(arm),'features'].iloc[0])}
            for k,metric in enumerate(['srocc','plcc','rmse']):
                finite=np.isfinite(boot[arm][:,k])
                row.update({metric:values[k],metric+'_low':np.nanpercentile(boot[arm][:,k],2.5),metric+'_high':np.nanpercentile(boot[arm][:,k],97.5),metric+'_invalid':int((~finite).sum())})
            summaries.append(row)
        pair_rows=[]
        if not paired and arm_column == 'variant':
            for arm in arms:
                reference = 'pc3_views_012345' if arm.startswith('pc3_') else 'views_012345'
                if arm == reference:
                    continue
                observed = scalar_metrics(y,arms[arm])-scalar_metrics(y,arms[reference])
                for k,metric in enumerate(['srocc','plcc','rmse']):
                    samples = boot[arm][:,k]-boot[reference][:,k]
                    tests.append({'dataset':dataset,'variant':arm,'reference':reference,'metric':metric,'difference_a_minus_b':observed[k],'low':np.nanpercentile(samples,2.5),'high':np.nanpercentile(samples,97.5),'bootstrap_invalid':int((~np.isfinite(samples)).sum()),'resamples':REPEATS})
        if paired:
            for a,b in itertools.combinations(arms,2):
                print('Paired tests',dataset,a,b,flush=True)
                p,invalid=swapped_tests(y,arms[a],arms[b],refs)
                delta=scalar_metrics(y,arms[a])-scalar_metrics(y,arms[b])
                for k,metric in enumerate(['srocc','plcc','rmse']):
                    samples=boot[a][:,k]-boot[b][:,k]
                    pair_rows.append({'dataset':dataset,'arm_a':a,'arm_b':b,'metric':metric,'difference_a_minus_b':delta[k],'low':np.nanpercentile(samples,2.5),'high':np.nanpercentile(samples,97.5),'bootstrap_invalid':int((~np.isfinite(samples)).sum()),'p_value':p[k],'permutation_invalid':int(invalid[k]),'resamples':REPEATS,'correction_family':f'{dataset}: all {arm_column} pairs and three metrics'})
            corrected=holm([row['p_value'] for row in pair_rows])
            for row,value in zip(pair_rows,corrected):
                row['p_holm']=value
            tests.extend(pair_rows)
    return pd.DataFrame(summaries),pd.DataFrame(folds),pd.DataFrame(tests)


def feature_agreement():
    rows=[]
    dictionary=read('manifests','review_feature_dictionary.csv')
    names=dictionary[dictionary.family.eq('PointPCA+')].set_index('index').semantic.to_dict()
    for dataset in FAMILY_DATASETS:
        print('Feature agreement',dataset,flush=True)
        base=complete(dataset)
        matlab=predictors(dataset,'PointPCA+',base).to_numpy(float)
        rs=predictors(dataset,'PointPCA+RS',base).to_numpy(float)
        _,code=np.unique(base.REF,return_inverse=True)
        group_count=code.max()+1
        rng=np.random.default_rng(SEED)
        weights=rng.multinomial(group_count,np.full(group_count,1/group_count),size=REPEATS)[:,code]
        total=weights.sum(axis=1)
        for index in range(40):
            a=matlab[:,index];b=rs[:,index];error=b-a
            samples={'pearson':weighted_corr(a,b,weights),'spearman':weighted_corr(weighted_ranks(a,weights),weighted_ranks(b,weights),weights),'bias':weights@error/total,'mae':weights@np.abs(error)/total,'rmse':np.sqrt(weights@(error**2)/total)}
            observed={'pearson':weighted_corr(a,b,np.ones((1,len(a))))[0],'spearman':weighted_corr(rankdata(a),rankdata(b),np.ones((1,len(a))))[0],'bias':error.mean(),'mae':np.abs(error).mean(),'rmse':np.sqrt(np.mean(error**2))}
            row={'dataset':dataset,'index':index,'semantic':names[index],'error_direction':'RS minus MATLAB','constant_matlab':bool(np.ptp(a)==0),'constant_rs':bool(np.ptp(b)==0)}
            for name,values in samples.items():
                finite=np.isfinite(values)
                row[name]=observed[name]
                row[name+'_low']=np.percentile(values[finite],2.5) if finite.any() else np.nan
                row[name+'_high']=np.percentile(values[finite],97.5) if finite.any() else np.nan
                row[name+'_invalid']=int((~finite).sum())
            rows.append(row)
    write('summaries','review_feature_agreement.csv',pd.DataFrame(rows))


def statistics():
    audit()
    baseline=read('predictions','review_baselines.csv')
    standalone=baseline[baseline.method.eq('DISTS')]
    summary,folds,_=summarize_predictions(standalone,'method')
    write('summaries','review_projection_only.csv',summary)
    write('summaries','review_projection_only_folds.csv',folds)
    family=baseline[baseline.dataset.isin(FAMILY_DATASETS)&baseline.method.isin(METHODS)]
    summary,folds,tests=summarize_predictions(family,'method',paired=True)
    write('summaries','review_pca_family.csv',summary)
    write('summaries','review_pca_family_folds.csv',folds)
    write('summaries','review_pca_pairwise_tests.csv',tests)
    ablations=read('predictions','review_ablations.csv')
    iqa=ablations[ablations.experiment.eq('iqa')]
    summary,folds,tests=summarize_predictions(iqa,'variant',paired=True)
    write('summaries','review_iqa_complementarity.csv',summary)
    write('summaries','review_iqa_folds.csv',folds)
    write('summaries','review_iqa_pairwise_tests.csv',tests)
    summary,folds,intervals=summarize_predictions(ablations[~ablations.experiment.eq('iqa')],'variant')
    write('summaries','review_views_pooling_paired_intervals.csv',intervals)
    write('summaries','review_views_pooling.csv',summary)
    write('summaries','review_views_pooling_folds.csv',folds)
    feature_agreement()
    print('Statistics complete',flush=True)


def tex(value):
    if isinstance(value,float) and not np.isfinite(value):
        return '--'
    s=str(value)
    return ''.join({'_':r'\_', '&':r'\&', '%':r'\%', '#':r'\#', '$':r'\$', '{':r'\{', '}':r'\}', '\\':r'\textbackslash{}'}.get(c,c) for c in s)


def label(value):
    return 'PointPCA$^3$' if value=='PointPCA3' else tex(value)


def design_label(value):
    names = {'pc3': 'PointPCA$^3$', 'dists': 'DISTS', 'topiq': 'TOPIQ-FR',
             'dists_topiq': 'DISTS + TOPIQ-FR', 'pc3_dists': 'PointPCA$^3$ + DISTS',
             'pc3_topiq': 'PointPCA$^3$ + TOPIQ-FR',
             'pc3_dists_topiq': 'PointPCA$^3$ + DISTS + TOPIQ-FR'}
    if value in names:
        return names[value]
    if 'pool_' in value:
        return ('PointPCA$^3$ + ' if value.startswith('pc3_') else '') + 'DISTS (' + value.split('pool_')[1] + ')'
    raise ValueError(value)


def number(value):
    return '--' if not np.isfinite(float(value)) else f'{value:.3f}'


def feature_error(value):
    if 0 < abs(value) < .0005:
        coefficient,exponent=f'{value:.1e}'.split('e')
        return coefficient+'e'+str(int(exponent))
    return number(value)


def table(name,caption,ref,headers,rows,spec=None,wide=True):
    environment='table*' if wide else 'table'
    width=r'\textwidth' if wide else r'\columnwidth'
    spec=spec or ('l'+'r'*(len(headers)-1))
    content=[r'\begin{'+environment+'}[t!]',r'\centering',r'\caption{'+caption+'}',r'\label{'+ref+'}',r'\scriptsize',r'\begin{adjustbox}{max width='+width+'}',r'\begin{tabular}{'+spec+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    content += [' & '.join(row)+r' \\' for row in rows]
    content += [r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{'+environment+'}']
    path('tables',name).write_text('\n'.join(content)+'\n')


def registry_report():
    selection=json.loads(path('manifests','core3_selection.json').read_text())
    roster=json.loads(path('manifests','run_manifest.json').read_text())['lazy_regressors']
    checkpoints=pd.concat([pd.read_csv(p) for p in sorted(path('checkpoints','core3_regressor').glob('*.csv'))],ignore_index=True)
    assert len(checkpoints)==41*75
    rows=[];parameters={}
    model_classes=B.model_class_map()
    for model in roster:
        cls=model_classes[model]
        estimator=cls()
        params=estimator.get_params()
        if 'random_state' in params:
            estimator.set_params(random_state=SEED)
        parameters[model]=estimator.get_params(deep=True)
        eligible=model in selection['regressor_details']['eligible_models']
        for dataset in ['APSIPA','WPC','LS-PCQA']:
            part=checkpoints[checkpoints.model.eq(model)&checkpoints.dataset.eq(dataset)]
            assert len(part)==25
            valid=np.isfinite(part[['srocc','plcc','rmse']].to_numpy()).all(axis=1)
            reason='eligible in all recorded inner folds' if eligible else ('sanity baseline excluded from selection' if model=='DummyRegressor' else 'excluded by recorded finite-metric eligibility; at least one inner metric undefined')
            rows.append({'model':model,'dataset':dataset,'eligible':eligible,'reason':reason,'checkpoint_statuses':','.join(sorted(part.status.unique())),'valid_metric_folds':int(valid.sum()),'recorded_folds':len(part),'builtin_cv':model.endswith('CV'),'builtin_cv_grouping':'estimator default; not supplied reference groups' if model.endswith('CV') else 'not applicable',**{metric:part[metric].median() for metric in ['srocc','plcc','rmse']}})
    frame=pd.DataFrame(rows)
    write('summaries','review_regressors.csv',frame)
    save_json('manifests','review_regressor_parameters.json',{'parameter_source':'reconstructed class defaults in version-matched numerical environment, random_state=42 where supported; no new fitting/search','parameter_limitations':'The original registry does not record XGBoost version or fitted parameters for every candidate. Reconstructed defaults are not a recovered per-fit parameter snapshot. Recorded checkpoint metrics are retained unchanged.','current_xgboost_version':importlib.metadata.version('xgboost'),'final_pipeline':json.loads(path('manifests','final_pointpcapp_run_manifest.json').read_text())['resolved_model_pipeline'],'all_models':parameters})
    latex=[]
    for model in roster:
        part=frame[frame.model.eq(model)].set_index('dataset')
        row=[tex(model), 'E' if part.eligible.all() else 'S' if model=='DummyRegressor' else 'U']
        for dataset in ['APSIPA','WPC','LS-PCQA']:
            row += [number(part.loc[dataset,metric]) for metric in ['srocc','plcc','rmse']]
        latex.append(row)
    table('review_regressors.tex',r'All 41 recorded regressors on the 46-feature representation. Each dataset triplet is SROCC/PLCC/RMSE, summarized as the median over its 25 nested inner-validation checkpoints. E: globally eligible; S: excluded sanity baseline; U: at least one undefined inner metric. A completed fit can still have undefined correlation. No explicit hyperparameter search was performed; CV estimators retain their built-in defaults. These selection summaries are not independent test results.','tab_review_regressors',['Regressor','Status',r'\multicolumn{3}{c}{APSIPA}',r'\multicolumn{3}{c}{WPC}',r'\multicolumn{3}{c}{LS-PCQA}'],latex,spec='ll'+'r'*9)
    p=path('tables','review_regressors.tex');s=p.read_text();s=s.replace(r'\midrule',' & & '+' & '.join(['SROCC','PLCC','RMSE']*3)+r' \\'+'\n'+r'\midrule',1);p.write_text(s)


def literature_report():
    # Historical provenance only; these unmatched results must not generate manuscript tables.
    # Keep combined cells and literal dataset IDs in the preserved evidence.
    bmsb_methods=['PointPCA3','PointPCA+RS','PointPCA+','PointPCA','PCQM','pSSIM','BitDance','Plane2plane','P2P MSE','PSNR Y']
    bmsb_values={'D0':[.937,.938,.942,.941,.940,.925,.859,.847,.896,.798],'D2':[.955,.956,.957,.890,.862,.708,.748,.761,.612,.743],'D3':[.910,.901,.909,.866,.749,.465,.451,.454,.563,.614]}
    sbrt_methods=['PointPCA+/RS (combined)','PointPCA','PCQM','pSSIM','BitDance','Plane2plane','P2Plane MSE','P2P MSE','PSNR Y']
    sbrt_values={'APSIPA':[.960,.941,.940,.925,.859,.847,.901,.896,.798],'SJTU-PCQA':[.980,.890,.862,.708,.748,.761,.578,.612,.743],'WPC':[.910,.866,.749,.465,.451,.454,.452,.563,.614]}
    rows=[]
    for dataset,values in bmsb_values.items():
        for method,value in zip(bmsb_methods,values):
            rows.append({'dataset':{'D0':'APSIPA','D2':'unresolved','D3':'WPC'}[dataset],'source_dataset_label':dataset,'method':method,'variant':'combined' if 'combined' in method else 'source-reported','source':'BMSB.pdf, Table III','result_type':'author-provided','protocol':'earlier-paper aggregate; exact population/split/calibration not established','sample_count':np.nan,'reference_count':np.nan,'srocc':value,'plcc':np.nan,'rmse':np.nan,'mapping_status':'quarantined' if dataset=='D2' else 'named by Table I; protocol unmatched','notes':'D2 conflicts with preceding D1/D2 ordering; no numeric reassignment' if dataset=='D2' else 'Source table identity retained; no paired inference'})
    for dataset,values in sbrt_values.items():
        for method,value in zip(sbrt_methods,values):
            rows.append({'dataset':dataset,'source_dataset_label':'M-PCCD' if dataset=='APSIPA' else dataset,'method':method,'variant':'combined' if 'combined' in method else 'source-reported','source':'PointPCA2-RS.pdf, Table V','result_type':'published','protocol':'earlier-paper aggregate; exact population/split/calibration not established','sample_count':np.nan,'reference_count':np.nan,'srocc':value,'plcc':np.nan,'rmse':np.nan,'mapping_status':'explicit dataset label; protocol unmatched','notes':'Combined PointPCA2/RS cell is not two independent results' if 'combined' in method else 'No paired inference with local OOF results'})
    frame=pd.DataFrame(rows)
    write('summaries','review_sota_evidence.csv',frame)
    write('audits','review_sota_quarantined.csv',frame[frame.mapping_status.eq('quarantined')])


def error_reports():
    frozen=read('predictions','final_pointpcapp_oof_predictions.csv')
    rows=[]
    for dataset,part in frozen.groupby('dataset',sort=False):
        meta=complete(dataset)[KEY+['CLASS']]
        part=part.merge(meta,on=KEY,validate='one_to_one')
        part['CLASS']=part['CLASS'].fillna('unknown').astype(str)
        part['residual']=part.y_pred-part.y_true
        part['absolute_error']=part.residual.abs()
        part['normalized_absolute_error']=part.absolute_error/(part.y_true.max()-part.y_true.min())
        rows.append(part)
    frame=pd.concat(rows,ignore_index=True)
    write('predictions','review_residuals.csv',frame)
    summaries=[]
    for column in ['REF','CLASS']:
        for (dataset,group),part in frame.groupby(['dataset',column],dropna=False,sort=True):
            values=scalar_metrics(part.y_true,part.y_pred)
            summaries.append({'dataset':dataset,'grouping':column,'group':group,'samples':len(part),'bias':part.residual.mean(),'mae':part.absolute_error.mean(),'normalized_mae':part.normalized_absolute_error.mean(),**dict(zip(['srocc','plcc','rmse'],values))})
    write('summaries','review_error_groups.csv',pd.DataFrame(summaries))
    strata=[]
    for dataset,part in frame.groupby('dataset',sort=False):
        lower,upper=part.y_true.quantile([1/3,2/3])
        for name,group in [('lower tercile',part[part.y_true<=lower]),('upper tercile',part[part.y_true>=upper])]:
            strata.append({'dataset':dataset,'stratum':name,'lower_cutoff':lower,'upper_cutoff':upper,'boundary_policy':'include ties at both tercile cutoffs','samples':len(group),'references':group.REF.nunique(),'bias':group.residual.mean(),'mae':group.absolute_error.mean(),'rmse':np.sqrt(np.mean(group.residual**2))})
    write('summaries','review_score_strata.csv',pd.DataFrame(strata))
    cases=[]
    for dataset,part in frame.groupby('dataset',sort=False):
        ordered=part.sort_values(['absolute_error',*KEY],kind='stable')
        for kind,subset in [('lowest error',ordered.head(2)),('highest error',ordered.tail(2))]:
            subset=subset.copy();subset['case_type']=kind;cases.append(subset)
    cases=pd.concat(cases,ignore_index=True)
    write('summaries','review_cases.csv',cases)
    latex=[]
    for dataset,part in cases.groupby('dataset',sort=False):
        for kind in ['lowest error','highest error']:
            selected=part[part.case_type.eq(kind)]
            row=selected.iloc[0] if kind=='lowest error' else selected.iloc[-1]
            latex.append([tex(dataset),tex(kind),tex(row.REF),number(row.y_true),number(row.y_pred),number(row.residual)])
    table('review_cases.tex',r'Deterministic illustrative lowest/highest absolute-error OOF predictions per dataset for the frozen PointPCA++ model. Residual is prediction minus MOS. These examples illustrate individual cases, not subgroup performance; the saved case artifact also retains the second example and full distorted-sample identity for each category.','tab_review_cases',['Dataset','Case','Reference','MOS','Prediction','Residual'],latex)
    return frame


def figures(residuals):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    import seaborn as sns
    from matplotlib.colors import TwoSlopeNorm
    sns.set_theme(context='paper',style='whitegrid',palette='deep',font='DejaVu Sans')
    summary=read('summaries','projection_case_summary.csv')
    pipelines=list(B.PIPELINES)
    grid=summary.pivot(index='pipeline',columns='iqa',values='srocc').loc[pipelines,B.IQAS]
    assert grid.size==220 and np.isfinite(grid).all().all()
    midpoint=.5
    span=max(abs(grid.to_numpy().min()-midpoint),abs(grid.to_numpy().max()-midpoint))
    # Ten columns leave room for all five characters at manuscript size.
    display=grid.T
    fig,ax=plt.subplots(figsize=(9,8.3))
    sns.heatmap(display,cmap='RdBu',norm=TwoSlopeNorm(vmin=midpoint-span,vcenter=midpoint,vmax=midpoint+span),annot=True,fmt='.3f',annot_kws={'color':'white','size':10.2},linewidths=.25,cbar_kws={'label':'Median grouped-fold SROCC'},ax=ax)
    for artist in ax.texts:
        artist.set_path_effects([effects.withStroke(linewidth=.55,foreground='#202020')])
    pipeline_labels = {'crop': 'Crop', 'nocrop': 'No crop', 'fsr': 'FSRFast', 'navier': 'Navier–Stokes', 'telea': 'Telea', 'shiftmap': 'ShiftMap', 'nopad': 'No padding'}
    ax.set_xticklabels(['\n'.join([pipeline_labels[x.split('_')[0]], pipeline_labels[x.split('_')[-1]].replace('Navier–Stokes','Navier–\nStokes').replace('No padding','No\npadding').replace('ShiftMap','Shift\nMap')]) for x in display.columns],rotation=0,fontsize=10.2)
    ax.set_yticklabels([x.replace('Metric','').replace('_','-') for x in display.index],rotation=0,fontsize=10)
    ax.set_xlabel('Projection pipeline');ax.set_ylabel('IQA')
    ax.tick_params(labelsize=10.5)
    ax.xaxis.label.set_size(10.5);ax.yaxis.label.set_size(10.5)
    fig.axes[-1].tick_params(labelsize=10.5);fig.axes[-1].yaxis.label.set_size(10.5)
    fig.tight_layout();fig.savefig(path('figures','review_projection_heatmap.pdf'));plt.close(fig)
    save_json('manifests','review_heatmap.json',{'cells':220,'annotations':[{'pipeline':p,'iqa':i,'value':float(grid.loc[p,i]),'display':f'{grid.loc[p,i]:.3f}'} for p in grid.index for i in grid.columns],'midpoint':midpoint,'vmin':midpoint-span,'vmax':midpoint+span,'midpoint_meaning':'display convention, not a significance or quality threshold','colormap':'RdBu','text':'white with dark outline'})
    sns.set_context('paper',font_scale=1.3)
    fig,axes=plt.subplots(2,4,figsize=(10.5,6.2))
    for ax,dataset in zip(axes.flat,DATASETS):
        part=residuals[residuals.dataset.eq(dataset)]
        ax.scatter(part.y_true,part.y_pred,s=8,alpha=.45,rasterized=True)
        limits=[min(part.y_true.min(),part.y_pred.min()),max(part.y_true.max(),part.y_pred.max())]
        ax.plot(limits,limits,color='gray',ls='--',lw=1)
        ax.set_title(dataset);ax.set_xlabel('MOS');ax.set_ylabel('OOF prediction')
        ax.set_xlim(limits);ax.set_ylim(limits)
    axes.flat[-1].axis('off');fig.tight_layout();fig.savefig(path('figures','review_mos_scatter.pdf'));plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(10.5,6.2))
    for ax,dataset in zip(axes.flat,DATASETS):
        part=residuals[residuals.dataset.eq(dataset)]
        ax.scatter(part.y_true,part.residual,s=8,alpha=.4,rasterized=True)
        limits=[min(part.y_true.min(),part.y_pred.min()),max(part.y_true.max(),part.y_pred.max())]
        ax.set_xlim(limits)
        ax.axhline(0,color='gray',ls='--',lw=1)
        ax.set_title(dataset);ax.set_xlabel('MOS');ax.set_ylabel('Prediction minus MOS')
    axes.flat[-1].axis('off');fig.tight_layout();fig.savefig(path('figures','review_residuals.pdf'));plt.close(fig)
    from review_presentation import descriptor_label
    agreement=read('summaries','review_feature_agreement.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),sharex=True,sharey=True)
    for ax,dataset in zip(axes.flat,FAMILY_DATASETS):
        part=agreement[agreement.dataset.eq(dataset)].sort_values('index')
        ax.axvspan(-.5,11.5,color='#eaf0f6',zorder=0)
        ax.axvline(11.5,color='0.65',ls=':',lw=1)
        ax.text(5.5,1.065,'Texture',ha='center',fontsize=11)
        ax.text(25.5,1.065,'Geometry',ha='center',fontsize=11)
        ax.scatter(part['index']-.12,part.pearson,marker='o',s=14,label='Pearson',zorder=3)
        ax.vlines(part['index']+.12,part.spearman_low,part.spearman_high,color='#dd8452',lw=.8,zorder=2)
        ax.scatter(part['index']+.12,part.spearman,marker='s',s=12,color='#dd8452',label='Spearman (95% CI)',zorder=3)
        worst=part.loc[part.pearson.idxmin()]
        ax.annotate(f'{int(worst["index"])}: {descriptor_label(worst.semantic)}',xy=(worst['index']-.12,worst.pearson),xytext=(.03,.2),textcoords='axes fraction',fontsize=11,arrowprops={'arrowstyle':'-','lw':.6,'color':'0.4'})
        ax.set_title(dataset,pad=30);ax.set_xlabel('Corresponding feature index');ax.set_ylabel('PointPCA+ and PointPCA+RS\ncorrelation')
        ax.set_ylim(0,1.035);ax.set_xlim(-1,40);ax.set_xticks(range(0,40,5))
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=11,bbox_to_anchor=(.5,-.08),frameon=False)
    fig.tight_layout(h_pad=2.5);fig.savefig(path('figures','review_feature_agreement.pdf'),bbox_inches='tight');plt.close(fig)
    scaling=read('summaries','runtime_scaling_summary.csv');scaling=scaling[scaling.dataset.isin(DATASETS)]
    assert np.allclose(scaling.select_dtypes('number'), read('summaries','review_runtime_scaling.csv').select_dtypes('number'))
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for index,(dataset,part) in enumerate(scaling.groupby('dataset',sort=False)):
        marker=['o','s','^','D','v','P','X'][index]
        axes[0].plot(part.worker,part.median_time,marker=marker,ms=3,label=dataset)
        axes[1].plot(part.worker,part.median_speedup,marker=marker,ms=3)
    for ax in axes:
        ax.axvline(16,color='gray',ls=':',lw=1);ax.set_xlabel('Workers (log scale)');ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],['1','2','4','8','16','32'])
    axes[0].set_yscale('log');axes[0].set_ylabel('Median extraction time (s)');axes[1].set_ylabel('Speedup relative to one worker')

    for ax in axes:
        ax.tick_params(which='both',labelsize=12.3);ax.xaxis.label.set_size(12.3);ax.yaxis.label.set_size(12.3)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=12.3,bbox_to_anchor=(.5,-.12));fig.tight_layout();fig.savefig(path('figures','review_worker_scaling.pdf'),bbox_inches='tight');plt.close(fig)


def quality_tables():
    projection=read('summaries','review_projection_only.csv')
    rows=[]
    for row in projection.itertuples():
        rows.append([tex(row.dataset)]+[f'{getattr(row,m):.3f} [{getattr(row,m+"_low"):.3f}, {getattr(row,m+"_high"):.3f}]' for m in ['srocc','plcc','rmse']])
    table('review_projection_only.tex',r'Standalone six-view DISTS prediction using the final complete-file projection inputs, fixed Extra Trees pipeline, and persisted five reference-disjoint folds. Values are pooled OOF metrics with reference-clustered 95\% intervals (10,000 draws, seed 42). Dataset populations and roles are in Table~\ref{tab_datasets}. RMSE is dataset-specific.','tab_review_projection_only',['Dataset','SROCC','PLCC','RMSE'],rows)
    full=read('summaries','projection_case_summary.csv');full=full[full.pipeline.eq('crop_pad_navier')]
    rows=[[tex(r.iqa.replace('Metric','')),f'{r.srocc:.3f} [{r.srocc_iqr:.3f}]',f'{r.plcc:.3f} [{r.plcc_iqr:.3f}]',f'{r.rmse:.3f} [{r.rmse_iqr:.3f}]',f'{r.median_time:.3f}'] for r in full.itertuples()]
    table('review_projection_iqas.tex',r'All 22 IQAs for the fixed crop/Navier--Stokes pipeline on APSIPA. Accuracy is median [IQR] across five grouped folds with the original fixed Extra Trees estimator; time is the recorded median projection/IQA interval in seconds. This historical extraction differs from the final complete-file DISTS experiment.','tab_review_projection_iqas',['IQA','SROCC [IQR]','PLCC [IQR]','RMSE [IQR]','Time (s)'],rows)
    family=read('summaries','review_pca_family.csv')
    rows=[]
    for dataset in FAMILY_DATASETS:
        row=[tex(dataset)]
        for method in METHODS:
            part=family[family.dataset.eq(dataset)&family.method.eq(method)]
            row += [number(part[m].iloc[0]) if len(part) else '--' for m in ['srocc','plcc','rmse']]
        rows.append(row)
    headers=['Dataset']+[r'\multicolumn{3}{c}{'+label(method)+'}' for method in METHODS]
    table('review_pca_family.tex',r'Controlled PCA-family prediction on complete matched datasets. Each method triplet is SROCC/PLCC/RMSE from pooled OOF predictions, using the same five content-disjoint cross-validation folds and selected Extra Trees pipeline. Original PointPCA uses its native 46 predictors on APSIPA; -- denotes extraction that could not be completed because memory requirements exceeded the available 128 GB RAM. PointPCA+, PointPCA+RS, and PointPCA$^3$ use 40 predictors; PointPCA++ uses 46. Populations/roles are in Table~\ref{tab_datasets}; intervals and paired inference accompany the saved summaries. These are common-regressor comparisons, not reproductions of each method\textquotesingle s native regression protocol.','tab_review_pca_family',headers,rows,spec='l'+'r'*15)
    # Explicit second header gives the metric order under every grouped method.
    p=path('tables','review_pca_family.tex');s=p.read_text();s=s.replace(r'\midrule', ' & '+' & '.join(['SROCC','PLCC','RMSE']*5)+r' \\'+'\n'+r'\midrule',1);p.write_text(s)
    tests=read('summaries','review_pca_pairwise_tests.csv')
    focused=tests[(tests.arm_a.eq('PointPCA+')&tests.arm_b.eq('PointPCA+RS'))|(tests.arm_a.eq('PointPCA++')&tests.arm_b.eq('PointPCA+'))]
    rows=[]
    for r in focused.itertuples():
        reverse=r.arm_a=='PointPCA+'
        arm_a,arm_b=(r.arm_b,r.arm_a) if reverse else (r.arm_a,r.arm_b)
        delta,low,high=(-r.difference_a_minus_b,-r.high,-r.low) if reverse else (r.difference_a_minus_b,r.low,r.high)
        rows.append([tex(r.dataset),label(arm_a)+' / '+label(arm_b),r.metric.upper(),f'{feature_error(delta)} [{feature_error(low)}, {feature_error(high)}]',f'{r.p_holm:.3f}'])
    table('review_pairwise.tex',r'Focused paired comparisons on shared OOF samples. Differences are first minus second method: PointPCA+RS minus PointPCA+, or PointPCA++ minus PointPCA+. Positive SROCC/PLCC and negative RMSE favor the first. Brackets are unadjusted, pointwise paired reference-clustered 95\% intervals. Two-sided reference-level prediction-label-swap p-values use 10,000 draws and a plus-one correction, followed by Holm correction over all available method pairs and all three metrics within each dataset (30 tests on APSIPA; 18 elsewhere). These adjusted tests need not agree with exclusion of zero by a pointwise bootstrap interval. Nonsignificance is not equivalence.','tab_review_pairwise',['Dataset','Methods','Metric',r'Difference [95\% CI]',r'$p_{\mathrm{Holm}}$'],rows)
    iqas=read('summaries','review_iqa_complementarity.csv')
    rows=[[design_label(r.variant),str(r.features)]+[f'{getattr(r,m):.3f} [{getattr(r,m+"_low"):.3f}, {getattr(r,m+"_high"):.3f}]' for m in ['srocc','plcc','rmse']] for r in iqas.itertuples()]
    table('review_iqa_complementarity.tex',r'Diagnostic DISTS/TOPIQ-FR combinations on APSIPA using the historical standalone IQA extraction, fixed Extra Trees pipeline, and persisted final grouped folds. Plus signs indicate concatenation. Metrics are pooled OOF values with reference-clustered 95\% intervals. Extraction cost is unmeasured for the combined designs; these comparisons do not reopen final selection.','tab_review_iqa_complementarity',['Features','Dimension','SROCC','PLCC','RMSE'],rows)
    views=read('summaries','review_views_pooling.csv')
    rows=[]
    for prefix in ['views_','pc3_views_']:
        for size in [2,4,6]:
            part=views[views.variant.str.startswith(prefix)&views.variant.str[len(prefix):].str.len().eq(size)]
            assert len(part)=={2:15,4:15,6:1}[size]
            rows.append(['DISTS' if prefix=='views_' else 'PointPCA$^3$ + DISTS',str(size),str(len(part)),str(int(part.features.iloc[0])),f'{part.srocc.median():.3f} [{part.srocc.min():.3f}, {part.srocc.max():.3f}]',f'{part.plcc.median():.3f}',f'{part.rmse.median():.3f}'])
    for r in views[views.variant.str.contains('pool_')].itertuples():
        rows.append([design_label(r.variant),'6','1',str(r.features),number(r.srocc),number(r.plcc),number(r.rmse)])
    table('review_views_pooling.tex',r'APSIPA diagnostic view-count and pooling comparisons with fixed Extra Trees and persisted grouped folds. The 2-/4-view rows summarize all 15 indexed subsets: SROCC is the median [minimum, maximum] across subsets, not a confidence interval; PLCC/RMSE are subset medians. Pooling uses all six DISTS views. No opposing-camera interpretation or measured speedup is inferred. The six-view concatenation is the reference; the final design remains fixed.','tab_review_views_pooling',['Representation','Views','Subsets','Dimension','SROCC [range]','PLCC','RMSE'],rows)
    agreement=read('summaries','review_feature_agreement.csv');names=read('manifests','review_feature_dictionary.csv').query("family == 'PointPCA+'").set_index('index').semantic
    rows=[]
    for index in range(40):
        row=[str(index),tex(names[index])]
        for dataset in FAMILY_DATASETS:
            r=agreement[agreement.dataset.eq(dataset)&agreement['index'].eq(index)].iloc[0]
            row += [number(r.spearman),feature_error(r.mae)]
        rows.append(row)
    table('review_feature_agreement.tex',r'Per-feature agreement between PointPCA+ and PointPCA+RS on matched samples. Each dataset pair is Spearman correlation and MAE in that feature\textquotesingle s native units; nonzero MAE below 0.0005 uses compact scientific notation. Indices follow the audited 40-slot correspondence; different features need not share units. Pearson correlation, bias, RMSE, clustered intervals and invalid/constant-value flags are retained in the accompanying numerical artifact. Agreement is descriptive and does not establish equivalence.','tab_review_feature_agreement',['Index','Feature',r'\multicolumn{2}{c}{APSIPA}',r'\multicolumn{2}{c}{ICIP2023}',r'\multicolumn{2}{c}{SJTU-PCQA}',r'\multicolumn{2}{c}{UnB\_PC}'],rows,spec='ll'+'r'*8)
    p=path('tables','review_feature_agreement.tex');s=p.read_text();s=s.replace(r'\midrule',' & & '+' & '.join(['Spearman','MAE']*4)+r' \\'+'\n'+r'\midrule',1);p.write_text(s)


def setup_and_runtime_tables(presentation_only=False):
    published={'APSIPA':('8','254','Color','G-PCC, V-PCC',r'\cite{apsipa_dataset}'),'ICIP2023':('75','1494','Color, normals','Compression',r'\cite{basics_dataset}'),'LS-PCQA':('104',r'$>22{,}000$','Color','Noise, resampling, compression',r'\cite{liu2023lspcqa}'),'SJTU-PCQA':('9','378','Color','Octree, resampling, noise',r'\cite{yang2020predicting}'),'UnB_PC':('6','60','Color, normals','Octree, JPEG',r'\cite{Torlig2018anovel}'),'WPC':('20','740','Color','Noise, resampling, G/V-PCC',r'\cite{wpc}'),'WPC2':('16','400','Color','V-PCC',r'\cite{liu2021wpc2}')}
    rows=[];inventory=[]
    for i,dataset in enumerate(DATASETS,1):
        base=complete(dataset);refs,samples,attributes,distortions,citation=published[dataset]
        role='Modelling' if dataset in ['APSIPA','WPC','LS-PCQA'] else 'Evaluation'
        subset='Yes' if dataset in FAMILY_DATASETS else '--'
        score_range=f'{base.SCORE.min():.2f}--{base.SCORE.max():.2f}'
        rows.append([str(i),tex(dataset)+' '+citation,tex(attributes),tex(distortions),refs,samples,str(base.REF.nunique()),str(len(base)),score_range,role,subset])
        inventory.append({'id':i,'dataset':dataset,'attributes':attributes,'distortions':distortions,'published_references':refs,'published_samples':samples,'analyzed_references':base.REF.nunique(),'analyzed_samples':len(base),'score_min':base.SCORE.min(),'score_max':base.SCORE.max(),'role':role,'family_comparison':dataset in FAMILY_DATASETS,'notes':'LS-PCQA uses the supplied accurate-MOS subset; larger published corpus includes generated scores' if dataset=='LS-PCQA' else 'Original PointPCA coverage is APSIPA only' if dataset=='APSIPA' else 'WPC2 class labels missing; shared WPC contents' if dataset=='WPC2' else ''})
    rows.append(['--','QoMEX2019','--','--','--','--',r'4$^\ast$',r'12$^\ast$','--','Setup only','--'])
    if not presentation_only:
        write('summaries','review_dataset_setup.csv',pd.DataFrame(inventory))
    table('review_dataset_setup.tex',r'Dataset properties and populations. Published and analyzed reference/sample counts are distinguished; the MOS column is the observed source-score range, retained without calibration or reversal. Modelling datasets influenced selection; evaluation datasets did not. Family identifies the four-dataset PCA-family study, whose populations equal the main analyzed populations; PointPCA is evaluated only on APSIPA because feature extraction for the other datasets exceeded the available 128 GB RAM. LS-PCQA uses 930 supplied accurate-MOS samples rather than its larger generated-score corpus. $^\ast$QoMEX2019 identifies 12 historical records/four references, excluded from every revised analysis; its original dataset identity remains unverified. WPC2 has unknown distortion-class metadata.','tab_datasets',['ID','Dataset','Attributes','Distortions','Pub. refs','Pub. samples','Used refs','Used samples','MOS range','Role','Family'],rows,spec='ll'+'l'*2+'r'*5+'ll')
    old=read('summaries','runtime_pointpca_summary.csv');filtered=old[old.dataset.isin(DATASETS)]
    if not presentation_only:
        write('summaries','review_runtime_pointpca.csv',filtered)
    lines=[]
    for r in filtered.itertuples():
        lines.append(f'{tex(r.dataset)} & {r.matlab_median_time:.3f} [{r.matlab_time_iqr:.3f}] & {r.pc3_w16_median_time:.3f} [{r.pc3_w16_time_iqr:.3f}] & {r.median_speedup:.2f} [{r.speedup_iqr:.2f}] '+r'\\')
    path('tables','review_runtime_pointpca_rows.tex').write_text('\n'.join(lines)+'\n'+r'\bottomrule'+'\n')
    # Preserve existing measured values and formatting while removing repeated counts.
    for source,dest,drop in [('tab_pointpcapp_all_datasets.tex','review_final_rows.tex',{1,2}),('runtime_pointpcapp_integrated_rows.tex','review_integrated_runtime_rows.tex',{1,2})]:
        file=path('tables',source)
        assert file.exists(), file
        lines=[]
        for line in file.read_text().splitlines():
            if '&' in line:
                cells=line.split('&');line=' & '.join(c.strip() for i,c in enumerate(cells) if i not in drop)
            lines.append(line)
        path('tables',dest).write_text('\n'.join(lines)+'\n')


def report_narratives():
    projection=read('summaries','review_projection_only.csv')
    pca=read('summaries','review_pca_family.csv')
    agreement=read('summaries','review_feature_agreement.csv')
    iqa=read('summaries','review_iqa_complementarity.csv').set_index('variant')
    residuals=read('predictions','review_residuals.csv')
    lines=[]
    projection_min=projection.loc[projection.srocc.idxmin()];projection_max=projection.loc[projection.srocc.idxmax()]
    lines.append(r'\newcommand{\ReviewProjectionInterpretation}{Table~\ref{tab_review_projection_only} evaluates the six DISTS scores independently of the point-cloud branch. Pooled SROCC ranges from '+f'{projection_min.srocc:.3f} on {tex(projection_min.dataset)} to {projection_max.srocc:.3f} on {tex(projection_max.dataset)}'+r'. This provides a standalone projection baseline on the same grouped folds as the fused model, rather than reusing fused predictions. Unlike the historical APSIPA IQA screening, this comparison uses the final extraction and pooled OOF metrics.}')
    diffs=[]
    for dataset in FAMILY_DATASETS:
        part=pca[pca.dataset.eq(dataset)].set_index('method')
        diffs.append(part.loc['PointPCA+RS','srocc']-part.loc['PointPCA+','srocc'])
    lines.append(r'\newcommand{\ReviewFamilyInterpretation}{Table~\ref{tab_review_pca_family} compares feature representations with the same Extra Trees pipeline and shared folds. The SROCC difference (PointPCA+RS minus PointPCA+) ranges from '+f'{min(diffs):.3f} to {max(diffs):.3f}'+r'. PointPCA++ has the strongest SROCC on APSIPA, ICIP2023, and SJTU-PCQA among these controlled methods, while PointPCA$^3$ is slightly higher on UnB\_PC. PointPCA+ has lower RMSE on ICIP2023 than PointPCA++. Thus, the fusion does not dominate every method on every metric. Original PointPCA is evaluated only on APSIPA using its native 46-feature vector. These comparisons isolate representations under a common regressor and do not reproduce each historical method\textquotesingle s native regression procedure.}')
    paired=read('summaries','review_pca_pairwise_tests.csv')
    rs=paired[paired.arm_a.eq('PointPCA+') & paired.arm_b.eq('PointPCA+RS') & paired.metric.eq('srocc')].set_index('dataset')
    r=rs.loc['SJTU-PCQA']
    agreement_example=f' On SJTU-PCQA, for example, PointPCA+RS minus PointPCA+ SROCC is {feature_error(-r.difference_a_minus_b)} with a pointwise 95\\% interval [{feature_error(-r.high)}, {feature_error(-r.low)}]. '
    lines.append(r'\newcommand{\ReviewAgreementInterpretation}{Figure~\ref{fig_review_feature_agreement} and Table~\ref{tab_review_feature_agreement} compare corresponding PointPCA+ and PointPCA+RS predictors after verifying their intended descriptor order. Across the 160 feature--dataset pairs, median Spearman correlation is '+f'{agreement.spearman.median():.3f}'+r', with a minimum of '+f'{agreement.spearman.min():.3f}'+r'. Per-feature errors reveal departures that a single pooled correlation would conceal.'+agreement_example+r'Table~\ref{tab_review_pairwise} reports paired prediction differences and Holm-adjusted tests; none of the paired comparisons between PointPCA+ and PointPCA+RS is significant after the declared correction. The tests do not establish equivalence; the estimated differences and their intervals quantify the agreement supported by these populations. Inspected source establishes intended feature semantics, while exact historical native binary versions remain a provenance limitation.}')
    a=iqa.loc['pc3_dists'];b=iqa.loc['pc3_dists_topiq']
    paired_iqa=read('summaries','review_iqa_pairwise_tests.csv')
    paired_iqa=paired_iqa[paired_iqa.arm_a.eq('pc3_dists') & paired_iqa.arm_b.eq('pc3_dists_topiq') & paired_iqa.metric.eq('srocc')].iloc[0]
    lines.append(r'\newcommand{\ReviewIQAInterpretation}{Table~\ref{tab_review_iqa_complementarity} tests whether TOPIQ-FR adds predictive information to DISTS on APSIPA. Adding its six views to the PointPCA$^3$/DISTS representation changes pooled SROCC from '+f'{a.srocc:.3f} to {b.srocc:.3f}'+r' and RMSE from '+f'{a.rmse:.3f} to {b.rmse:.3f}'+r'. The paired SROCC change is '+f'{-paired_iqa.difference_a_minus_b:.3f}'+r' (pointwise 95\% interval ['+f'{-paired_iqa.high:.3f}, {-paired_iqa.low:.3f}'+r'], Holm-adjusted $p='+f'{paired_iqa.p_holm:.3f}'+r'$), so the lower point estimate is not a demonstrated performance loss. Combining the two IQAs without PointPCA$^3$ also lowers the SROCC point estimate relative to DISTS alone. High importance in a larger correlated feature set therefore does not imply a benefit from this concatenation. These diagnostic results retain the selected 46-feature design and do not establish behavior beyond APSIPA.}')
    views=read('summaries','review_views_pooling.csv').set_index('variant')
    four=views[views.index.str.startswith('pc3_views_') & (views.index.str.len() == len('pc3_views_')+4)]
    pooled=views[views.index.str.startswith('pc3_pool_')]
    lines.append(r'\newcommand{\ReviewViewInterpretation}{For the fused representation, the median four-view SROCC is '+f'{four.srocc.median():.3f}'+r', compared with '+f'{views.loc["pc3_views_012345","srocc"]:.3f}'+r' for six views. Across the four pooling functions, fused SROCC ranges from '+f'{pooled.srocc.min():.3f} to {pooled.srocc.max():.3f}'+r'. Pooling therefore reduces the input dimension from 46 to 41 with lower point estimates in this dataset, while some indexed subsets approach the six-view estimate. Paired reference-clustered intervals against each six-view baseline are retained for every design in the numerical results; these descriptive comparisons do not establish a universally preferable subset or pooling rule.}')
    ma=[]
    for dataset,part in residuals.groupby('dataset',sort=False):
        ma.append((dataset,part.normalized_absolute_error.mean()))
    worst=max(ma,key=lambda x:x[1])
    strata=read('summaries','review_score_strata.csv')
    low=strata[strata.stratum.eq('lower tercile')].set_index('dataset')
    high=strata[strata.stratum.eq('upper tercile')].set_index('dataset')
    compressed=int(((low.bias>0)&(high.bias<0)).sum())
    lines.append(r'\newcommand{\ReviewErrorInterpretation}{Table~\ref{tab_review_error_summary} quantifies dataset-wise errors; '+tex(worst[0])+r' has the largest mean absolute error relative to its observed MOS range ('+f'{worst[1]:.3f}'+r'). In '+('all seven datasets' if compressed==7 else str(compressed)+' of the seven datasets')+r', mean residuals are positive in the lowest MOS tercile and negative in the highest, consistent with compression toward central scores. On WPC2 these biases are '+f'{low.loc["WPC2","bias"]:.3f} and {high.loc["WPC2","bias"]:.3f}'+r', respectively. The terciles include ties at their empirical boundaries and are descriptive, post-hoc score strata. Tables~\ref{tab_review_cases} and~\ref{tab_review_error_groups} give deterministic examples and the strongest recorded class-level error concentrations. Unknown WPC2 classes remain unknown, and small or constant-score subgroups need not have defined correlation. These observations identify error concentrations without establishing their causes.}')
    path('tables','review_macros.tex').write_text('\n'.join(lines)+'\n')
    error_summary=[]
    for dataset,part in residuals.groupby('dataset',sort=False):
        error_summary.append([tex(dataset),number(part.residual.mean()),number(part.absolute_error.mean()),number(part.normalized_absolute_error.mean())])
    table('review_error_summary.tex',r'Frozen PointPCA++ OOF prediction errors. Bias is prediction minus MOS; normalized MAE divides each sample\textquotesingle s absolute error by its dataset\textquotesingle s observed MOS range. Normalization is descriptive and does not calibrate predictions. Populations and scales are in Table~\ref{tab_datasets}.','tab_review_error_summary',['Dataset','Bias','MAE','Normalized MAE'],error_summary)
    groups=read('summaries','review_error_groups.csv');rows=[]
    for dataset in DATASETS:
        part=groups[groups.dataset.eq(dataset)&groups.grouping.eq('CLASS')].sort_values(['mae','group'],ascending=[False,True]).head(2)
        for r in part.itertuples():
            rows.append([tex(dataset),tex(r.group),number(r.bias),number(r.mae),number(r.rmse)])
    table('review_error_groups.tex',r'The two recorded distortion-class identifiers with greatest MAE per dataset (or all available classes if fewer). Grouping follows supplied CLASS metadata; no semantic class names are inferred. Unknown WPC2 labels form one unknown group. These post-hoc error concentrations are descriptive; complete class/reference summaries retain group sizes in the numerical artifacts.','tab_review_error_groups',['Dataset','Class identifier','Bias','MAE','RMSE'],rows)


def reports():
    audit()
    registry_report()
    literature_report()
    residuals=error_reports()
    quality_tables()
    setup_and_runtime_tables()
    report_narratives()
    architecture_figures()
    write('summaries','review_runtime_scaling.csv',read('summaries','runtime_scaling_summary.csv').query('dataset in @DATASETS'))
    __import__('review_presentation').render(__import__(__name__))
    response_matrix()
    print('Reports complete',flush=True)


def architecture_figures():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    def draw(name, boxes, arrows, size):
        fig,ax=plt.subplots(figsize=size)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
        for x,y,w,h,text in boxes:
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor='#e8eff8',edgecolor='#4c72b0',linewidth=1))
            ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
        for a,b in arrows:
            ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1,'color':'#404040'})
        fig.tight_layout(pad=.3);fig.savefig(path('figures',name));plt.close(fig)
    draw('review_rs_architecture.pdf',[(.02,.25,.16,.5,'Reference and\ndistorted clouds'),(.23,.25,.17,.5,'Merge duplicates;\nbuild KD-trees'),(.45,.25,.17,.5,'On-demand KNN;\nlocal PCA/descriptors'),(.67,.25,.15,.5,'Mean-pool\n40 predictors'),(.87,.25,.11,.5,'Regression')],[((.18,.5),(.23,.5)),((.4,.5),(.45,.5)),((.62,.5),(.67,.5)),((.82,.5),(.87,.5))],(12,1.5))
    draw('review_pc3_architecture.pdf',[(.02,.25,.17,.5,'Reference and\ndistorted clouds'),(.24,.25,.19,.5,'Duplicate merging;\nKD-trees and VoSS'),(.48,.25,.21,.5,'Seed neighborhoods;\nPCA and descriptors'),(.74,.25,.24,.5,'Mean-pool 40\nPointPCA³ predictors')],[((.19,.5),(.24,.5)),((.43,.5),(.48,.5)),((.69,.5),(.74,.5))],(11,1.5))
    draw('review_pointpcapp_architecture.pdf',[(.02,.35,.17,.3,'Reference and\ndistorted clouds'),(.26,.62,.29,.25,'PointPCA³\n40 predictors'),(.26,.10,.29,.35,'Six orthographic projections\nCrop + Navier–Stokes + DISTS\n6 view scores'),(.63,.35,.15,.3,'Concatenate\n46 features'),(.84,.35,.14,.3,'Extra Trees\nquality prediction')],[((.19,.5),(.26,.75)),((.19,.5),(.26,.28)),((.55,.75),(.63,.5)),((.55,.28),(.63,.5)),((.78,.5),(.84,.5))],(11,3.2))
    draw('review_orthographic_schematic.pdf',[(.02,.27,.20,.46,'Shared point-cloud\npreprocessing'),(.29,.27,.20,.46,'Six perpendicular\northographic views'),(.56,.27,.18,.46,'Occupancy-based\ncropping/inpainting'),(.81,.27,.17,.46,'Six paired\nIQA scores')],[((.22,.5),(.29,.5)),((.49,.5),(.56,.5)),((.74,.5),(.81,.5))],(10,1.6))


REQUESTS = [
    (1,'Projection-only results and discussion','tab_review_projection_only;tab_review_projection_iqas','summaries/review_projection_only.csv;summaries/projection_case_summary.csv',''),
    (2,'Numeric heatmap cells','fig_projection_pipeline_heatmap','manifests/review_heatmap.json',''),
    (3,'Diverging heatmap and white text','fig_projection_pipeline_heatmap','figures/review_projection_heatmap.pdf;manifests/review_heatmap.json',''),
    (4,'SOTA/baseline comparison','sec_results_pointpcapp_sota','summaries/review_sota_evidence.csv;audits/review_sota_quarantined.csv','Same-protocol SOTA comparison results are not available; earlier-paper aggregates are retained only as provenance and excluded from manuscript analyses'),
    (5,'PCA-family SROCC/PLCC/RMSE table','tab_review_pca_family','summaries/review_pca_family.csv;audits/review_inputs.csv',''),
    (6,'Per-feature agreement and statistical evidence','fig_review_feature_agreement;tab_review_feature_agreement;tab_review_pairwise','summaries/review_feature_agreement.csv;summaries/review_pca_pairwise_tests.csv;manifests/review_feature_dictionary.csv',''),
    (7,'Standalone PointPCA+ versus PointPCA+RS on shared folds','tab_review_pca_family;tab_review_pairwise','predictions/review_baselines.csv;summaries/review_pca_pairwise_tests.csv',''),
    (8,'Efficiency including PointPCA+RS/hybrid SOTA','sec_results_pointpcapp_runtime','summaries/review_runtime_pointpca.csv;summaries/runtime_pointpcapp_integrated_summary.csv','Comparable JVCI PointPCA+RS/hybrid timings and independently aligned timer boundaries not supplied; historical supplement timing excluded'),
    (9,'Prediction versus MOS scatterplots','fig_review_mos_scatter','predictions/final_pointpcapp_oof_predictions.csv;figures/review_mos_scatter.pdf',''),
    (10,'Success/failure analysis','fig_review_residuals;tab_review_error_summary;tab_review_cases;tab_review_error_groups','predictions/review_residuals.csv;summaries/review_error_groups.csv;summaries/review_cases.csv',''),
    (11,'Resolution ablation','sec_results_diagnostics','manifests/review_method_provenance.json','Raw APSIPA inputs/reproducible extraction or audited multi-resolution feature package missing'),
    (12,'View-count accuracy and cost ablation','tab_review_views_pooling','predictions/review_ablations.csv;summaries/review_views_pooling.csv;summaries/review_views_pooling_paired_intervals.csv','Accuracy complete; changed-configuration extraction cost unavailable; historical camera mapping unverified'),
    (13,'View-pooling accuracy and cost ablation','tab_review_views_pooling','summaries/review_views_pooling.csv;summaries/review_views_pooling_paired_intervals.csv','Accuracy complete; changed-configuration extraction cost unavailable'),
    (14,'DISTS plus TOPIQ-FR complementarity','tab_review_iqa_complementarity','summaries/review_iqa_complementarity.csv;summaries/review_iqa_pairwise_tests.csv','Accuracy and paired inference complete; dual-IQA extraction cost unavailable'),
    (15,'Hyperparameter/default transparency','tab_review_regressors;sec_setup','manifests/review_regressor_parameters.json;manifests/core3_selection.json',''),
    (16,'All 41 regressors','tab_review_regressors','summaries/review_regressors.csv;checkpoints/core3_regressor',''),
    (17,'Method terminology and lineage','sec_pointpca_rs;fig_arquitecture_pointpca3','manifests/review_method_provenance.json;manifests/review_feature_dictionary.csv',''),
    (18,'QoMEX exclusion and enumeration','tab_datasets;fig_pointpca3_worker_scaling','summaries/review_dataset_setup.csv;summaries/review_runtime_scaling.csv',''),
    (19,'One dataset-property/population setup table','tab_datasets','summaries/review_dataset_setup.csv;tables/review_dataset_setup.tex',''),
]


def response_matrix(validated=False):
    rows=[]
    for identifier,request,labels,evidence,dependency in REQUESTS:
        missing=[item for item in evidence.split(';') if not (OUT/item).exists()]
        status='blocked' if dependency or missing else 'complete' if validated else 'in progress'
        rows.append({'id':identifier,'request':request,'status':status,'implemented_component':'Unmatched literature table removed; red pending-comparison note at subsection footer; historical evidence retained only as provenance' if identifier==4 else 'Matched JVCI PointPCA+ timing, scope and claim corrections' if identifier==8 else 'Resolution provenance and explicit missing-input record only' if identifier==11 else 'Available accuracy/statistical evidence and explicit unmeasured cost' if identifier in [12,13,14] else 'Requested available-data implementation','manuscript_labels':labels,'evidence':';'.join('data/analysis/'+item for item in evidence.split(';')),'checks':'source preservation, available evidence existence, manuscript label/reference checks; experiment-specific counts/folds/metrics in audits/review_validation.json' if validated else 'pending validation','dependency':dependency or ';'.join(missing)})
    frame=pd.DataFrame(rows)
    write('manifests','review_response_matrix.csv',frame)
    return frame


def provenance():
    sources={
        'review_projection_only':['summaries/review_projection_only.csv'],
        'review_projection_iqas':['summaries/projection_case_summary.csv'],
        'review_pca_family':['summaries/review_pca_family.csv'],
        'review_pairwise':['summaries/review_pca_pairwise_tests.csv'],
        'review_iqa_complementarity':['summaries/review_iqa_complementarity.csv','summaries/review_iqa_pairwise_tests.csv'],
        'review_views_pooling':['summaries/review_views_pooling.csv','summaries/review_views_pooling_paired_intervals.csv'],
        'review_feature_agreement':['summaries/review_feature_agreement.csv'],
        'review_mos_scatter':['predictions/final_pointpcapp_oof_predictions.csv'],
        'review_residuals':['predictions/review_residuals.csv'],
        'review_error_summary':['predictions/review_residuals.csv'],
        'review_error_groups':['summaries/review_error_groups.csv'],
        'review_cases':['summaries/review_cases.csv'],
        'review_regressors':['summaries/review_regressors.csv'],
        'review_dataset_setup':['summaries/review_dataset_setup.csv'],
        'review_projection_heatmap':['summaries/projection_case_summary.csv','manifests/review_heatmap.json'],
        'review_worker_scaling':['summaries/review_runtime_scaling.csv'],
        'review_pointpcapp_runtime':['predictions/runtime_pointpcapp_integrated_by_sample.csv'],
        'review_pointpcapp_runtime_speedup':['predictions/runtime_pointpcapp_integrated_by_sample.csv'],
        'review_feature_importance':['summaries/importance_individual.csv','summaries/importance_iqa_blocks.csv','manifests/review_feature_dictionary.csv'],
        'review_projection_accuracy_runtime':['summaries/projection_pipeline_ranks.csv'],
        'review_regressor_comparison':['summaries/core3_regressor_inner.csv','summaries/core3_regressor_dataset_ranks.csv','summaries/core3_regressor_selection_summary.csv'],
        'review_feature_set_ablation':['summaries/core3_feature_set_fold_metrics.csv'],
        'review_single_iqa_selection':['summaries/single_iqa_outer_selection.csv'],
        'review_pointpcapp_modality_importance':['summaries/cross_dataset_modality_importance_summary.csv'],
        'review_pointpcapp_all_datasets':['summaries/cross_dataset_transfer_summary.csv'],
        'review_runtime_pointpca_rows':['summaries/review_runtime_pointpca.csv'],
        'review_integrated_runtime_rows':['summaries/runtime_pointpcapp_integrated_summary.csv'],
        'review_final_rows':['summaries/final_pointpcapp_dataset_summary.csv'],
        'review_macros':['summaries/review_projection_only.csv','summaries/review_pca_family.csv','summaries/review_feature_agreement.csv','summaries/review_iqa_complementarity.csv','summaries/review_iqa_pairwise_tests.csv','summaries/review_views_pooling.csv','summaries/review_score_strata.csv','predictions/review_residuals.csv'],
    }
    rows=[]
    for directory in ['tables','figures']:
        for file in sorted(path(directory,'').glob('review_*')):
            if not file.is_file():
                continue
            inputs=sources.get(file.stem,['manifests/review_method_provenance.json'])
            rows.append({'artifact':str(file.relative_to(ROOT)),'data_sources':';'.join('data/analysis/'+p for p in inputs),'source_hashes':';'.join(sha(OUT/p) for p in inputs),'artifact_sha256':sha(file),'run_id':'review_20261009','fields':'all displayed values derive from named inputs; architecture diagrams are qualitative'})
    write('manifests','review_artifact_provenance.csv',pd.DataFrame(rows))


def validate():
    audit()
    for filename,arm in [('review_baselines.csv','method'),('review_ablations.csv','variant')]:
        frame=read('predictions',filename)
        assert np.isfinite(frame[['y_true','y_pred']].to_numpy()).all()
        assert not frame.duplicated(['dataset',arm,*KEY]).any()
        for (dataset,method),part in frame.groupby(['dataset',arm]):
            base=complete(dataset)
            assert len(part)==len(base)
            expected=base[KEY].copy();expected['fold']=assignments(dataset,base);expected['y_true']=base.SCORE.to_numpy()
            joined=expected.merge(part,on=KEY,validate='one_to_one',suffixes=('_expected',''))
            assert np.array_equal(joined.fold_expected,joined.fold)
            np.testing.assert_allclose(joined.y_true_expected,joined.y_true,rtol=0,atol=1e-14)
    frozen=read('predictions','final_pointpcapp_oof_predictions.csv')
    baselines=read('predictions','review_baselines.csv')
    comparison=frozen.merge(baselines[baselines.method.eq('PointPCA++')],on=['dataset',*KEY,'fold'],suffixes=('_frozen',''),validate='one_to_one')
    assert len(comparison)==3638
    np.testing.assert_allclose(comparison.y_pred_frozen,comparison.y_pred,rtol=0,atol=1e-14)
    family=read('summaries','review_pca_family.csv')
    for row in family.itertuples():
        part=baselines[baselines.dataset.eq(row.dataset)&baselines.method.eq(row.method)]
        np.testing.assert_allclose(scalar_metrics(part.y_true,part.y_pred),[row.srocc,row.plcc,row.rmse],rtol=0,atol=1e-12)
    tests=read('summaries','review_pca_pairwise_tests.csv')
    assert tests.groupby('dataset').size().to_dict()=={'APSIPA':30,'ICIP2023':18,'SJTU-PCQA':18,'UnB_PC':18}
    for _,part in tests.groupby('dataset'):
        np.testing.assert_allclose(holm(part.p_value),part.p_holm,atol=1e-14)
        assert part.resamples.eq(REPEATS).all()
    iqa=read('summaries','review_iqa_pairwise_tests.csv');assert len(iqa)==63
    np.testing.assert_allclose(holm(iqa.p_value),iqa.p_holm,atol=1e-14)
    assert len(read('summaries','review_feature_agreement.csv'))==160
    registry=read('summaries','review_regressors.csv');assert registry.model.nunique()==41 and len(registry)==123
    assert registry[registry.eligible].model.nunique()==34
    assert len(read('audits','review_sota_quarantined.csv'))==10
    heatmap=json.loads(path('manifests','review_heatmap.json').read_text())
    scores=read('summaries','projection_case_summary.csv').set_index(['pipeline','iqa']).srocc
    assert len(heatmap['annotations'])==220
    for cell in heatmap['annotations']:
        assert cell['display']==f"{scores.loc[(cell['pipeline'],cell['iqa'])]:.3f}"
        assert heatmap['vmin']<=cell['value']<=heatmap['vmax']
    for filename in ['review_runtime_pointpca.csv','review_runtime_scaling.csv','review_dataset_setup.csv']:
        assert set(read('summaries',filename).dataset)==set(DATASETS)
    import re
    manuscript=(ROOT/'main.tex').read_text()
    def expand(text):
        text=re.sub(r'(?<!\\)%[^\n]*','',text)
        return re.sub(r'\\input\{([^}]+)\}',lambda m:expand((ROOT/m.group(1)).read_text()),text)
    expanded=expand(manuscript)
    labels=re.findall(r'\\label\{([^}]+)\}',expanded)
    assert len(labels)==len(set(labels)), 'duplicate labels'
    references=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',expanded)
    assert set(references)<=set(labels),set(references)-set(labels)
    body=re.sub(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}','',expanded,flags=re.S)
    exhibits=set(re.findall(r'\\label\{((?:fig|tab)[^}]+)\}',expanded))
    body_references=set(re.findall(r'\\ref\{([^}]+)\}',body))
    assert exhibits<=body_references, exhibits-body_references
    for filename in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',expanded):
        assert (ROOT/filename).is_file(),filename
    for _,_,required,_,_ in REQUESTS:
        for item in required.split(';'):
            assert item in labels,item
    assert 'Author TODO' not in expanded
    assert not re.search(r'&\s*(?:Samples|References)\s*&',expanded)
    assert 'QoMEX' not in path('tables','review_runtime_pointpca_rows.tex').read_text()
    provenance()
    matrix=response_matrix(validated=True)
    output={'source_csvs_verified':300,'frozen_artifacts_verified':len(json.loads(path('manifests','review_revision_manifest.json').read_text())['frozen_artifact_hashes']),'frozen_prediction_rows':3638,'baseline_prediction_rows':len(baselines),'diagnostic_prediction_rows':len(read('predictions','review_ablations.csv')),'diagnostic_arms':read('predictions','review_ablations.csv').variant.nunique(),'feature_dataset_pairs':160,'heatmap_cells':220,'registry_entries':41,'exhibits_referenced_outside_captions':len(exhibits),'request_status_counts':matrix.status.value_counts().to_dict(),'full_review_complete':False,'errors':[]}
    output['reading_flow'] = __import__('review_presentation').validate_structure(__import__(__name__))
    save_json('audits','review_validation.json',output)
    print(json.dumps(output,indent=2),flush=True)
