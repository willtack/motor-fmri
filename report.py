#!/usr/bin/env python
#
#   nipype script to analyze fmri runs preprocessed by fmriprep, compute statistics related to language lateralization
#   and compile these results into an html report
#
#   Intended for clinical language task fMRIs of presurgicalEpilepsy patients
#
#   Will Tackett, University of Pennsylvania
#
#   August 7th, 2019
#

from __future__ import division
from __future__ import print_function

import glob
import json
import os
import sys
from builtins import range
from builtins import str

import matplotlib.pyplot as plt
import nibabel
import nilearn.plotting
import nipype.algorithms.modelgen as model  # model generation
import nipype.interfaces.ants as ants  # fsl
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pipeline engine
import numpy as np
import pandas as pd
from bids import BIDSLayout
from nipype.interfaces.base import Bunch

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
datadir = '/home/will/report_gear/'
bidsdir = datadir + 'bids_dataset'

# try:
#     bidsdir = sys.argv[1]
# except IOError:
#     print("No input file specified.")
#     sys.exit(1)

layout = BIDSLayout(bidsdir)


lba_mask = datadir + "masks/lba.nii.gz"
rba_mask = datadir + "masks/rba.nii.gz"
lstg_mask = datadir + "masks/stg_left.nii.gz"
rstg_mask = datadir + "masks/stg_right.nii.gz"
lmtg_mask = datadir + "masks/mtg_left.nii.gz"
rmtg_mask = datadir + "masks/mtg_right.nii.gz"
litg_mask = datadir + "masks/itg_left.nii.gz"
ritg_mask = datadir + "masks/itg_right.nii.gz"
lsfg_mask = datadir + "masks/sfg_left.nii.gz"
rsfg_mask = datadir + "masks/sfg_right.nii.gz"
lwa_mask = datadir + "masks/stg_post_left.nii.gz"
rwa_mask = datadir + "masks/stg_post_right.nii.gz"
lhem_mask = datadir + "masks/hem_left.nii.gz"
rhem_mask = datadir + "masks/hem_right.nii.gz"

mtl_mask = datadir + "masks/hpf_bin.nii.gz"
lmtl_mask = datadir + "masks/hpf_left_bin.nii.gz"
rmtl_mask = datadir + "masks/hpf_right_bin.nii.gz"
lhc_mask = datadir + "masks/hippocampus_left_bin.nii.gz"
rhc_mask = datadir + "masks/hippocampus_right_bin.nii.gz"
lffg_mask = datadir + "masks/ffg_left_bin.nii.gz"
rffg_mask = datadir + "masks/ffg_right_bin.nii.gz"
lphg_mask = datadir + "masks/phg_left_bin.nii.gz"
rphg_mask = datadir + "masks/phg_right_bin.nii.gz"


def setup(taskname):
    events = pd.read_csv(os.path.join(bidsdir, "task-" + taskname + "_events.tsv"), sep="\t")
    source_img = layout.get(task=taskname, run=1, session="01", suffix="bold", extension="nii.gz")[0]
    print('Using ' + source_img.filename + ' as source image.')

    subject_info = [Bunch(conditions=[taskname],
                          onsets=[list(events[events.trial_type == 'stimulus'].onset)],
                          durations=[list(events[events.trial_type == 'stimulus'].duration)])]

    prepped_img = os.path.join(bidsdir, "derivatives",
                               "fmriprep", "sub-" + source_img.entities['subject'],
                               "ses-" + source_img.entities['session'], "func",
                               "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                   'session'] + "_task-" + taskname + "_run-01_space-MNI152NLin2009cAsym_desc"
                                                                      "-smoothAROMAnonaggr_bold.nii.gz")
    return source_img, prepped_img, subject_info


def model_fitting(source_img, prepped_img, subject_info, task):

    taskdir = datadir + task + '/'
    if not os.path.exists(taskdir):
        os.mkdir(taskdir)

    # skull strip the preprocessed BOLD
    bet = fsl.BET()
    bet.inputs.in_file = prepped_img
    bet.inputs.frac = 0.7
    bet.inputs.functional = True
    bet.inputs.out_file = taskdir + task + "_input_functional_bet.nii.gz"
    bet_res = bet.run()
    bettedinput = bet_res.outputs.out_file

    task_vs_baseline = [task + " vs baseline", 'T', [task], [1]]  # set up contrasts
    contrasts = [task_vs_baseline]

    modelfit = pe.Workflow(name='modelfit', base_dir=taskdir)  # generate the model fitting workflow
    modelspec = pe.Node(interface=model.SpecifyModel(), name="modelspec")  # generate design info
    level1design = pe.Node(interface=fsl.Level1Design(), name="level1design")  # generate fsf file
    modelgen = pe.MapNode(  # generate .mat file
        interface=fsl.FEATModel(),
        name='modelgen',
        iterfield=['fsf_file', 'ev_files'])
    feat = pe.Node(  # feat statistics
        interface=fsl.FEAT(),
        name='feat',
        iterfield=['fsf_file'])

    # put it all together
    modelfit.connect([
        (modelspec, level1design, [('session_info', 'session_info')]),
        (level1design, modelgen, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
        (level1design, feat, [('fsf_files', 'fsf_file')])])

    # define inputs to workflow
    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.functional_runs = bettedinput
    modelspec.inputs.time_repetition = source_img.entities['RepetitionTime']
    modelspec.inputs.high_pass_filter_cutoff = 90
    modelspec.inputs.subject_info = subject_info

    level1design.inputs.interscan_interval = source_img.entities['RepetitionTime']
    level1design.inputs.bases = {'gamma': {'gammasigma': 3, 'gammadelay': 6, 'derivs': True}}
    level1design.inputs.contrasts = contrasts
    level1design.inputs.model_serial_correlations = True

    # Run the model-fitting pipeline. Main outputs are a feat directory (w/ functional img) and a design.mat file
    res = modelfit.run()

    # outputs
    feat_dir = list(res.nodes)[3].result.outputs.feat_dir
    thresh_img = feat_dir + "/thresh_zstat1.nii.gz"

    return thresh_img


# noinspection PyTypeChecker
def post_stats(img, task):

    taskdir = datadir + task + '/'
    if not os.path.exists(taskdir):
        os.mkdir(taskdir)

    # save a glass brain image, threshold of z > 6
    if task == 'scenemem':
        # masked_img = fsl.ImageMaths(in_file=img, mask_file=mtl_mask, out_file=taskdir + task + "_img_masked.nii.gz")
        nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(img, 8),
                                          output_file=taskdir + task + "_gb.svg",
                                          display_mode='lyrz', colorbar=True, plot_abs=False, threshold=5)
    else:
        nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(img, 8), output_file=taskdir + task + "_gb.svg",
                                          display_mode='lyrz', colorbar=True, plot_abs=False, threshold=3.5)

    def get_mask_vox(msk):
        mask_stat = fsl.ImageStats(in_file=msk, op_string=' -V')
        mask_run = mask_stat.run()
        mask_vox = list(mask_run.outputs.out_stat)
        return mask_vox[0]

    def get_roi_perc(image, msk, mask_vox):
        roi_stat = fsl.ImageStats(in_file=image, op_string='-k ' + msk + ' -V')
        print(roi_stat.cmdline)
        stat_run = roi_stat.run()
        stat = float(list(stat_run.outputs.out_stat)[0])
        perc = (stat / mask_vox) * 100
        return perc

    def calc_ar(left, right):
        return (left - right) / (left + right)

    vox = {}  # dictionary of mask: mask voxels
    res = {}  # dictionary of roi: percent activation
    left_stats = []  # for graph
    right_stats = []
    rois = []  # roi labels for graph
    ars = []  # list of asymmetry ratios for table

    if task == 'object':
        rois = ['whole brain', "broca's area"]
        masks = [lba_mask, rba_mask, lhem_mask, rhem_mask]
        for mask in masks:
            roi = os.path.basename(mask).split('.')[0]
            vox[roi + '_vox'] = get_mask_vox(mask)
            res[roi] = round(get_roi_perc(img, mask, vox[roi + '_vox']))
        left_stats = [res['hem_left'], res['lba']]
        right_stats = [res['hem_right'], res['rba']]
        ba_ar = round(calc_ar(res['lba'], res['rba']), 2)
        hem_ar = round(calc_ar(res['hem_left'], res['hem_right']), 2)
        ars = [hem_ar, ba_ar]
    elif task == 'rhyme':
        rois = ['whole brain', "broca's area", "wernicke's area"]
        masks = [lba_mask, rba_mask, lhem_mask, rhem_mask, lwa_mask, rwa_mask]
        for mask in masks:
            roi = os.path.basename(mask).split('.')[0]
            vox[roi + '_vox'] = get_mask_vox(mask)
            res[roi] = round(get_roi_perc(img, mask, vox[roi + '_vox']))
        left_stats = [res['hem_left'], res['lba'], res['stg_post_left']]
        right_stats = [res['hem_right'], res['rba'], res['stg_post_right']]
        ba_ar = round(calc_ar(res['lba'], res['rba']), 2)
        hem_ar = round(calc_ar(res['hem_left'], res['hem_right']), 2)
        wa_ar = round(calc_ar(res['stg_post_left'], res['stg_post_right']), 2)
        ars = [hem_ar, ba_ar, wa_ar]
    elif task == 'scenemem':
        rois = ['mTL', 'hippocampus', 'fusiform gyrus', 'parahippocampal gyrus']
        masks = [lmtl_mask, rmtl_mask, lhc_mask, rhc_mask, lffg_mask, rffg_mask, lphg_mask, rphg_mask]
        for mask in masks:
            roi = os.path.basename(mask).split('.')[0]
            vox[roi + '_vox'] = get_mask_vox(mask)
            res[roi] = round(get_roi_perc(img, mask, vox[roi + '_vox']))
        left_stats = [res['hpf_left_bin'], res['hippocampus_left_bin'], res['ffg_left_bin'], res['phg_left_bin']]
        right_stats = [res['hpf_right_bin'], res['hippocampus_right_bin'], res['ffg_right_bin'], res['phg_right_bin']]
        mtl_ar = round(calc_ar(res['hpf_left_bin'], res['hpf_right_bin']), 2)
        hc_ar = round(calc_ar(res['hippocampus_left_bin'], res['hippocampus_right_bin']), 2)
        ffg_ar = round(calc_ar(res['ffg_left_bin'], res['ffg_right_bin']), 2)
        phg_ar = round(calc_ar(res['phg_left_bin'], res['phg_right_bin']), 2)
        ars = [mtl_ar, hc_ar, ffg_ar, phg_ar]
    elif task == 'sentence':
        rois = ['wb', "ba", "superior TG", "middle TG", "inferior TG"]
        masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lstg_mask, rstg_mask, lmtg_mask, rmtg_mask, litg_mask, ritg_mask]
        for mask in masks:
            roi = os.path.basename(mask).split('.')[0]
            vox[roi + '_vox'] = get_mask_vox(mask)
            res[roi] = round(get_roi_perc(img, mask, vox[roi + '_vox']))
        left_stats = [res['hem_left'], res['lba'], res['stg_left'], res['mtg_left'], res['itg_left']]
        right_stats = [res['hem_right'], res['rba'], res['stg_right'], res['mtg_right'], res['itg_right']]
        ba_ar = round(calc_ar(res['lba'], res['rba']), 2)
        hem_ar = round(calc_ar(res['hem_left'], res['hem_right']), 2)
        stg_ar = round(calc_ar(res['stg_left'], res['stg_right']), 2)
        mtg_ar = round(calc_ar(res['mtg_left'], res['mtg_right']), 2)
        itg_ar = round(calc_ar(res['itg_left'], res['itg_right']), 2)
        ars = [hem_ar, ba_ar, stg_ar, mtg_ar, itg_ar]
    elif task == 'wordgen':
        rois = ['whole brain', "broca's area", "superior frontal gyrus"]
        masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lsfg_mask, rsfg_mask]
        for mask in masks:
            roi = os.path.basename(mask).split('.')[0]
            vox[roi + '_vox'] = get_mask_vox(mask)
            res[roi] = round(get_roi_perc(img, mask, vox[roi + '_vox']))
        left_stats = [res['hem_left'], res['lba'], res['sfg_left']]
        right_stats = [res['hem_right'], res['rba'], res['sfg_right']]
        ba_ar = round(calc_ar(res['lba'], res['rba']), 2)
        hem_ar = round(calc_ar(res['hem_left'], res['hem_right']), 2)
        sfg_ar = round(calc_ar(res['sfg_left'], res['sfg_right']), 2)
        ars = [hem_ar, ba_ar, sfg_ar]

    # Bar graph
    index = np.arange(len(left_stats))
    bar_width = 0.2
    opacity = 0.8
    axes = plt.gca()
    axes.set_ylim([0, 100])

    plt.bar(index, left_stats, bar_width,
            alpha=opacity,
            color='#4f6bb0',
            label='Left')
    plt.bar(index + bar_width, right_stats, bar_width,
            alpha=opacity,
            color='#550824',
            label='Right')

    plt.xlabel('ROI')
    plt.ylabel('% activated voxels in ROI')
    plt.title(task)
    plt.xticks(index + bar_width / 2, rois)
    plt.legend()
    plt.savefig(taskdir + task + '_bar.svg')
    plt.close()

    # generate statistics table
    row = rois
    column = ['left %', 'right %', 'asymmetry ratio']
    data = np.array([left_stats, right_stats, ars]).transpose()
    df = pd.DataFrame(data, index=row, columns=column)
    df_string = df.to_html().replace('<table border="1" class="dataframe">',
                                     '<table class="table table-bordered" style="width:45%" >').replace(
        '<tr style="text-align: right;">', '<tr style="text-align: left;">')

    with open(taskdir + task + "_table.txt", "w") as text_file:
        text_file.write(df_string)

    # interactive viewer
    html_view = nilearn.plotting.view_img(nilearn.image.smooth_img(img, 8), threshold=5, bg_img='MNI152', vmax=10, title=task)
    html_view.save_as_html(taskdir + task + '_viewer.html')


def generate_report(tasks):

    task_list = tasks
    gb = {}
    viewers = {}
    bars = {}
    tables = {}

    for tsk in task_list:
        taskdir = datadir + tsk + '/'
        gb[tsk + '_gb'] = taskdir + tsk + "_gb.svg"
        viewers[tsk + '_viewer'] = taskdir + tsk + "_viewer.html"
        bars[tsk + '_bar'] = taskdir + tsk + "_bar.svg"
        with open(taskdir + tsk + '_table.txt') as file:
            table_string = file.read()
        tables[tsk + '_table'] = table_string

    legend = '''<table class="table table-striped">
                <th>Key</th><th>Definition</th> 
                <tr>
                    <td>tSNR</td>
                    <td><img src="''' + datadir + '''imgs/tsnr_equation.png"</img>...where
                    <img src="''' + datadir + '''imgs/mean_signal.png"</img > is the average BOLD signal (across time), 
                    and <img src="''' + datadir + '''imgs/std_dev.png"</img>is the 
                    corresponding temporal standard-deviation map.</td> </tr>
                    <tr> <td>Left/Right %</td> <td>Activated voxels (p < 0.05, corr.) as a percentage of the specified 
                    ROI in the left/right side </td>
                </tr>
                <tr> 
                    <td>Asymmetry Ratio</td> <td>
                    <img src="''' + datadir + '''imgs/asym_ratio_equation.png"</img>
                    ...where “left activation” = “Left %” (as above)</td>
                     </tr>
                <tr> 
                <td>Mean FD</td>
                <td>The average frame-wise displacement, an expression of instantaneous head motion</a></td>
                </tr>
                </table> '''

    html_string = '''<?xml version="1.0" encoding="utf-8" ?> <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 
    Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"> <html 
    xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en"> <head> <meta http-equiv="Content-Type" 
    content="text/html; charset=utf-8" /> <meta name="generator" content="Docutils 0.12: 
    http://docutils.sourceforge.net/" /> <title></title> <script 
    src="https://code.jquery.com/jquery-3.3.1.slim.min.js" 
    integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" 
    crossorigin="anonymous"></script> <script 
    src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js" 
    integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" 
    crossorigin="anonymous"></script> <link rel="stylesheet" 
    href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css" 
    integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous"> 
    <link rel="stylesheet" href="styles.css">
    </head>
    <body>

    <nav class="navbar fixed-top navbar-expand-lg navbar-light bg-light">
    <div class="collapse navbar-collapse">
        <ul class="navbar-nav">
            <li class="nav-item"><a class="nav-link" href="#ses-01_task-object_run-01">Object Naming</a></li>
            <li class="nav-item"><a class="nav-link" href="#ses-01_task-rhyme_run-01">Rhyme Matching</a></li>
            <li class="nav-item"><a class="nav-link" href="#ses-01_task-scenemem_run-01">Scene Memory</a></li>
            <li class="nav-item"><a class="nav-link" href="#ses-01_task-sentence_run-01">Sentence Completion</a></li>
            <li class="nav-item"><a class="nav-link" href="#ses-01_task-wordgen_run-01">Wordgen</a></li>
        </ul>
    </div>
    </nav>



    <div id="Summary" style="top-padding: -20px">
        <h1 class="sub-report-title">Summary</h1>
    <ul class="elem-desc">
        <li>Subject ID: sub-P69</li>
          <li>Structural images: 1 T1-weighted </li>
          <li>Functional series: ''' + str(len(layout.get_tasks())) + '''</li>
               <ul class="elem-desc">
               <li>Task: ''' + layout.get_tasks()[0] + '''</li>
               <li>Task: ''' + layout.get_tasks()[1] + '''</li>
               <li>Task: ''' + layout.get_tasks()[2] + '''</li>
               <li>Task: ''' + layout.get_tasks()[3] + '''</li>
               <li>Task: ''' + layout.get_tasks()[4] + '''</li>
               <li>Task: ''' + layout.get_tasks()[5] + '''</li>
               </ul>
           <li>Resampling targets: MNI152NLin2009cAsym, T1w </li>
        </ul>
        <h3> Reference table </h3>
        ''' + legend + '''
    </div>

    <div id="Functional">
            <div id="ses-01_task-object_run-01" style="margin-top: 50px">
                <h1 class="run-title">Object Naming</h1>
                <h3 class="elem-title">Glass brain</h3>
                <div class="elem-image">
                            <object class="svg-reportlet" type="image/svg+xml" data="''' + gb['object_gb'] + '''">
                            </object>
                </div>

                <h3 style="margin-top: 20px"><a href=''' + viewers['object_viewer'] + '''>Statistical Map Viewer</a></h3>

                <h3 style="margin-top: 20px"> Statistics </h3>

                <div style="border: 1px solid black;"> 
                <div class="table">
                    ''' + tables['object_table'] + '''
                </div>
                <div class="barplot">
                            <object data="''' + bars['object_bar'] + '''">
                            </object>
                </div>
                </div>

            </div>
            
            <div id="ses-01_task-rhyme_run-01" style="margin-top: 75px">
                <h1 class="run-title">Rhyme Matching</h1>
                <h3 class="elem-title">Glass brain</h3>
                <div class="elem-image">
                            <object class="svg-reportlet" type="image/svg+xml" data="''' + gb['rhyme_gb'] + '''">
                            </object>
                </div>

                <h3 style="margin-top: 20px"><a href=''' + viewers['rhyme_viewer'] + '''>Statistical Map Viewer</a></h3>

                <h3 style="margin-top: 20px"> Statistics </h3>

                <div style="border: 1px solid black;"> 
                <div class="table">
                    ''' + tables['rhyme_table'] + '''
                </div>
                <div class="barplot">
                            <object data="''' + bars['rhyme_bar'] + '''">
                            </object>
                </div>
                </div>
                
                
            <div id="ses-01_task-scenemem_run-01" style="margin-top: 75px">
                <h1 class="run-title">Scene Memory</h1>
                <h3 class="elem-title">Glass brain</h3>
                <div class="elem-image">
                            <object class="svg-reportlet" type="image/svg+xml" data="''' + gb['scenemem_gb'] + '''">
                            </object>
                </div>

                <h3 style="margin-top: 20px"><a href=''' + viewers['scenemem_viewer'] + '''>Statistical Map Viewer</a></h3>

                <h3 style="margin-top: 20px"> Statistics </h3>

                <div style="border: 1px solid black;"> 
                <div class="table">
                    ''' + tables['scenemem_table'] + '''
                </div>
                <div class="barplot">
                            <object data="''' + bars['scenemem_bar'] + '''">
                            </object>
                </div>
            </div>       
                
            <div id="ses-01_task-sentence_run-01" style="margin-top: 75px">
                <h1 class="run-title">Sentence Completion</h1>
                <h3 class="elem-title">Glass brain</h3>
                <div class="elem-image">
                            <object class="svg-reportlet" type="image/svg+xml" data="''' + gb['sentence_gb'] + '''">
                            </object>
                </div>

                <h3 style="margin-top: 20px"><a href=''' + viewers['sentence_viewer'] + '''>Statistical Map Viewer</a></h3>

                <h3 style="margin-top: 20px"> Statistics </h3>

                <div style="border: 1px solid black;"> 
                <div class="table">
                    ''' + tables['sentence_table'] + '''
                </div>
                <div class="barplot">
                            <object data="''' + bars['sentence_bar'] + '''">
                            </object>
                </div>
            </div>
                
            <div id="ses-01_task-wordgen_run-01" style="margin-top: 75px">
                <h1 class="run-title">Word Generation</h1>
                <h3 class="elem-title">Glass brain</h3>
                <div class="elem-image">
                            <object class="svg-reportlet" type="image/svg+xml" data="''' + gb['wordgen_gb'] + '''">
                            </object>
                </div>

                <h3 style="margin-top: 20px"><a href=''' + viewers['wordgen_viewer'] + '''>Statistical Map Viewer</a></h3>

                <h3 style="margin-top: 20px"> Statistics </h3>

                <div style="border: 1px solid black;"> 
                <div class="table">
                    ''' + tables['wordgen_table'] + '''
                </div>
                <div class="barplot">
                            <object data="''' + bars['wordgen_bar'] + '''">
                            </object>
                </div>
            </div>

            </div>
    </div>
    </body>

    '''

    f = open(datadir + 'report.html', 'w')
    f.write(html_string)
    f.close()


def main():
    task_list = layout.get_tasks()
    task_list.remove('rest')

    for task in task_list:
        (source_epi, input_functional, info) = setup(task)
        thresholded_img = model_fitting(source_epi, input_functional, info, task)
        post_stats(thresholded_img, task)
    generate_report(task_list)


main()
