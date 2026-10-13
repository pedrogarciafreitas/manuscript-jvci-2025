"""Publication layouts from saved evidence; no fitting or statistical resampling."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd


def descriptor_label(semantic):
    """Readable names retain the audited descriptor identity and channel/axis."""
    simple = {'t_varsum': 'Color variance sum', 't_omnivariance': 'Color omnivariance',
              't_entropy': 'Color entropy', 'g_pB2pA': 'Point-to-point error'}
    if semantic in simple:
        return simple[semantic]
    for prefix, name in [('t_mu_', 'Color mean'), ('t_var_', 'Color variance'),
                         ('t_cov_', 'Color covariance')]:
        if semantic.startswith(prefix):
            return f'{name} ({semantic[len(prefix):].upper()})'
    geometry = {'g_varsum': 'Geometry variance sum', 'g_omnivariance': 'Geometry omnivariance',
                'g_entropy': 'Geometry eigenentropy', 'g_anisotropy': 'Anisotropy',
                'g_planarity': 'Planarity', 'g_linearity': 'Linearity',
                'g_sphericity': 'Scattering', 'g_curvature': 'Curvature', 'g_surfaceVariation': 'Surface variation'}
    if semantic in geometry:
        return geometry[semantic]
    for prefix,name in [('g_vAB2plA_', 'Error along axis'), ('g_pA2plA_', 'Reference point–plane'), ('g_pB2plA_', 'Test point–plane'), ('g_cB2plA_', 'Centroid–plane'), ('g_var_', 'Geometry variance'), ('g_cov_', 'Geometry covariance'), ('g_asim_', 'Angular similarity'), ('g_paralellity_', 'Parallelity')]:
        if semantic.startswith(prefix):
            return f'{name} ({semantic[len(prefix):].upper()})'
    return {'g_pB2cA':'Test point–centroid', 'g_cB2cA':'Centroid distance'}[semantic]


def render_importance(importance, dictionary, save):
    import matplotlib.pyplot as plt
    import seaborn as sns
    names = dictionary[dictionary.family.eq('PointPCA+')].set_index('index').semantic
    def readable(name):
        if re.fullmatch(r'pc3__?f[0-9]+', name):
            return descriptor_label(names[int(name.split('f')[-1])])
        if name.startswith('proj__'):
            iqa, view = name.removeprefix('proj__').split('__view')
            return f'{iqa.removesuffix('metric').upper().replace("_", "-")} (view {view})'
        return name.replace('Metric', '').replace('_', '-')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.7), gridspec_kw={'width_ratios': [1.2, 1]})
    for ax, level, title in zip(axes, ['individual', 'iqa_block'], ['Individual features', 'Six-view IQA blocks']):
        frame = importance[importance.level.eq(level)].nlargest(15, 'median_srocc_decrease').iloc[::-1]
        labels = [readable(x) for x in frame.feature]
        colors = ['#c44e52' if x == 'DISTS' else '#4c72b0' for x in labels]
        bars = ax.barh(np.arange(len(frame)), frame.median_srocc_decrease, color=colors)
        for label, bar in zip(labels, bars):
            if label == 'DISTS':
                bar.set_hatch('///'); bar.set_edgecolor('0.2')
        ax.set_yticks(np.arange(len(frame)), labels)
        values = frame.median_srocc_decrease.to_numpy()
        ax.bar_label(bars, labels=['0' if v == 0 else f'{v:.4f}' if abs(v)>=.00005 else f'{v:.1e}' for v in values], padding=3, fontsize=10)
        ax.set_xlim(0, max(values)*1.28)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel('Median held-out SROCC decrease')
        ax.set_title(title); ax.grid(axis='y', visible=False);sns.despine(ax=ax)
    fig.tight_layout(w_pad=2.5)
    save(fig, 'feature_importance.pdf')


def render_runtime(samples, save):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    datasets=['APSIPA','ICIP2023','LS-PCQA','SJTU-PCQA','UnB_PC','WPC','WPC2']
    assert len(samples)==3638
    assert np.isfinite(samples[['matlab_time','pointpcapp_time','speedup']]).all().all()
    assert (samples[['matlab_time','pointpcapp_time']]>0).all().all()
    for kind in ['time', 'speedup']:
        fig,axes=plt.subplots(2,4,figsize=(10.5,5.8))
        for i,(ax,dataset) in enumerate(zip(axes.flat,datasets)):
            part=samples[samples.dataset.eq(dataset)]
            x=part.total_points_millions
            if kind=='time':
                ax.scatter(x,part.matlab_time,s=12,c='#777777',marker='o',alpha=.5,rasterized=True)
                ax.scatter(x,part.pointpcapp_time,s=14,c='#4c72b0',marker='^',alpha=.5,rasterized=True)
                ax.set_yscale('log');ax.set_ylabel('Extraction time (s)')
            else:
                ax.scatter(x,part.speedup,s=12,c='#dd8452',marker='o',alpha=.5,rasterized=True)
                ax.set_ylim(bottom=0);ax.set_ylabel('Paired speedup (×)')
            ax.set_xlim(0,x.max()*1.04)
            ax.set_title(f'({chr(97+i)}) {dataset}')
            ax.set_xlabel('Total points (millions)');sns.despine(ax=ax)
        axes.flat[-1].axis('off')
        if kind=='time':
            handles=[Line2D([],[],color=c,marker=m,ls='',label=n) for c,m,n in [('#777777','o','MATLAB PointPCA+'),('#4c72b0','^','PointPCA++ (32 workers)')]]
            axes.flat[-1].legend(handles=handles,loc='center',fontsize=10)
        fig.tight_layout(h_pad=2,w_pad=1)
        save(fig,'pointpcapp_runtime.pdf' if kind=='time' else 'pointpcapp_runtime_speedup.pdf')


def protected_hashes(root):
    paths=list((root/'data/features').rglob('*.csv'))
    for folder in ['summaries','predictions','folds','checkpoints','normalized']:
        paths.extend(p for p in (root/'data/analysis'/folder).rglob('*') if p.is_file())
    manifest=json.loads((root/'data/analysis/manifests/review_revision_manifest.json').read_text())
    paths.extend(root/p for p in manifest['frozen_artifact_hashes'])
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def render(review):
    """Scoped entry point: only tables, figures, and presentation provenance change."""
    B=review.B
    before=protected_hashes(review.ROOT)
    review.quality_tables()
    review.report_narratives()
    review.figures(review.read('predictions','review_residuals.csv'))
    wanted={'projection_accuracy_runtime.pdf','single_iqa_selection.pdf',
            'regressor_comparison.pdf','feature_set_ablation.pdf','feature_importance.pdf',
            'pointpcapp_runtime.pdf','pointpcapp_runtime_speedup.pdf',
            'pointpcapp_modality_importance.pdf','pointpcapp_all_datasets.pdf'}
    import matplotlib.text as mtext
    old_save=B.save_figure
    def save(fig,filename):
        if filename not in wanted or (filename=='regressor_comparison.pdf' and len(fig.axes)!=4):
            B.plt.close(fig);return
        # Figure source fonts account for downscaling to the actual LaTeX width.
        width=3.30 if filename=='projection_accuracy_runtime.pdf' else 6.7
        minimum=8.0*fig.get_figwidth()/width
        for ax in fig.axes:
            ax.tick_params(axis='both',which='both',labelsize=minimum)
        for artist in fig.findobj(mtext.Text):
            if artist.get_text():
                artist.set_fontsize(max(artist.get_fontsize(),minimum))
        if filename=='projection_accuracy_runtime.pdf':
            fig.set_figheight(7.8)
            # Rebuild the legends: changing ncols alone does not repack them.
            legends=list(fig.legends)
            entries=[(list(leg.legend_handles),[t.get_text() for t in leg.get_texts()],leg.get_title().get_text()) for leg in legends]
            for leg in legends:
                leg.set_visible(False)
            for (handles,labels,title),ncol,y in zip(entries,[2,3],[.14,.005]):
                fig.legend(handles=handles,labels=labels,title=title,ncol=ncol,loc='lower center',bbox_to_anchor=(.5,y),fontsize=minimum,title_fontsize=minimum,frameon=False,handletextpad=.3,columnspacing=.8)
            fig._supxlabel.set_position((.5,.29))
            fig.subplots_adjust(bottom=.35,hspace=.3)
        if filename=='regressor_comparison.pdf':
            from matplotlib.ticker import MaxNLocator
            for ax in fig.axes:
                ax.xaxis.set_major_locator(MaxNLocator(3))
            for ax in fig.axes[:3]:
                ax.set_xlabel('Inner-validation\nSROCC')
            fig.axes[3].set_xlabel('Median aggregate\nrank')
        if filename=='pointpcapp_all_datasets.pdf':
            fig.tight_layout(rect=(0,.12,1,1))
        elif filename!='projection_accuracy_runtime.pdf':
            fig.tight_layout()
        fig.savefig(review.path('figures','review_'+filename),bbox_inches='tight',facecolor='white')
        B.plt.close(fig)
    try:
        B.save_figure=save
        reg=json.loads(review.path('manifests','regressor_selected.json').read_text())
        importance=pd.concat([review.read('summaries','importance_individual.csv'),review.read('summaries','importance_iqa_blocks.csv')],ignore_index=True)
        B.generate_figures(review.read('summaries','projection_case_summary.csv'),review.read('summaries','projection_pipeline_ranks.csv'),review.read('summaries','regressor_shortlist_outer.csv'),reg['stable_shortlist'],reg['selected_regressor'],importance,review.read('summaries','single_iqa_outer_selection.csv'),review.read('summaries','feature_set_ablation_summary.csv'),review.read('summaries','runtime_pointpca_summary.csv'),review.read('summaries','runtime_scaling_summary.csv'),review.read('summaries','runtime_pointpcapp_summary.csv'),B.load_pointpcapp_runtime_samples())
        B.render_core3_selection_figures()
        render_importance(importance,review.read('manifests','review_feature_dictionary.csv'),save)
        B.generate_cross_dataset_modality_importance_figure(review.read('summaries','cross_dataset_modality_importance_summary.csv'))
        B.generate_cross_dataset_transfer_figure(review.read('summaries','cross_dataset_transfer_summary.csv'))
    finally:
        B.save_figure=old_save
    after=protected_hashes(review.ROOT)
    assert before==after, 'Presentation regeneration changed protected evidence'
    review.provenance()
    review.save_json('audits','review_presentation_validation.json',{'protected_files':len(before),'protected_hashes':before,'numerical_evidence_unchanged':True,'models_refitted':False,'rendered_figures':sorted('review_'+x for x in wanted),'descriptor_label_provenance':{'dictionary':'data/analysis/manifests/review_feature_dictionary.csv','PointPCA3_dispatch':'c9a98f93ed62d63592de01de3a7a11b2e711a92f2bfde4d014738b44e6c68f59','shared_RS_predictors':'804f88fd2f0a2eddf22473523b8ebe77745e5cc926e2d2398067ebfceb8ee0b5','verification':'Inspected pinned PointPCA3 lib.rs delegates to the audited pointpca2-rs compute_predictors; v1.5.1 Cargo.lock resolves the same commit.'},'pdf_inspection':'pending current LaTeX build'})
    print(f'Presentation regenerated; {len(before)} source, numerical, and frozen artifacts unchanged',flush=True)


def validate_structure(review):
    """Guard the argument order and main-text coverage without interpreting numbers."""
    source=(review.ROOT/'main.tex').read_text()
    section=source.split(r'\section{Experimental Results}',1)[1].split(r'\section{Conclusion}',1)[0]
    responsibilities={
        'sec_results_projection_analysis': ['fig_projection_pipeline_heatmap','fig_projection_accuracy_runtime'],
        'sec_results_regression_features': ['fig_single_iqa_selection','fig_regressor_comparison','fig_feature_set_ablation'],
        'sec_results_pca_family': ['fig_review_feature_agreement','tab_review_pca_family'],
        'sec_results_pointpcapp_sota': ['tab_review_projection_only','tab_pointpcapp_all_datasets','tab_review_sota'],
        'sec_results_diagnostics': ['fig_feature_importance','tab_review_iqa_complementarity','tab_review_views_pooling'],
        'sec_results_errors': ['fig_pointpcapp_modality_importance','fig_review_mos_scatter','fig_review_residuals','tab_review_error_summary'],
        'sec_results_transfer': ['fig_pointpcapp_all_datasets'],
        'sec_results_pointpca3_runtime': ['fig_pointpca3_worker_scaling','tab_pointpca3_runtime','fig_pointpcapp_runtime','tab_pointpcapp_integrated_runtime'],
    }
    subsections=re.findall(r'\\subsection\{[^}]+\}\s*\\label\{([^}]+)\}',section)
    assert subsections==list(responsibilities),subsections
    chunks=re.split(r'\\subsection\{[^}]+\}\s*\\label\{[^}]+\}',section)[1:]
    for (name,expected),chunk in zip(responsibilities.items(),chunks):
        expanded=re.sub(r'\\input\{([^}]+)\}',lambda m:(review.ROOT/m[1]).read_text(),chunk)
        labels=re.findall(r'\\label\{((?:fig|tab)[^}]+)\}',expanded)
        assert set(labels)==set(expected),(name,labels,expected)
        assert chunk.rstrip().endswith(r'\FloatBarrier'),name
    freeze=section.index('These analyses fix the PointPCA++ configuration')
    assert section.index(r'\label{fig_feature_set_ablation}')<freeze<section.index(r'\label{sec_results_pca_family}')
    assert freeze<section.index(r'\input{data/analysis/tables/review_projection_only.tex}')
    assert r'\ContinuedFloat' in chunks[-1]
    assert 'RMSE' not in review.path('tables','review_error_summary.tex').read_text()
    assert '-0.000' not in review.path('tables','review_pairwise.tex').read_text()
    return {'subsection_order':subsections,'main_exhibits_retained':sum(map(len,responsibilities.values())),
            'selection_before_final_comparisons':True,'runtime_continued_parts':2,
            'float_barriers_at_argument_boundaries':True,'no_duplicate_error_rmse':True}
