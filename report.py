#!/usr/local/miniconda/bin/python
#
#   nipype script to analyze fmri runs preprocessed by fmriprep, compute statistics related to language lateralization
#   and compile these results into an html report
#
#   Intended for clinical language task fMRIs of presurgicalEpilepsy patients
#
#   Will Tackett, University of Pennsylvania
#
#   August 13th, 2019
#

from __future__ import division
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import nilearn.plotting
import nipype.algorithms.modelgen as model
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import numpy as np
import pandas as pd
from bids import BIDSLayout
from nipype.interfaces.base import Bunch
from jinja2 import FileSystemLoader, Environment
from nipype.algorithms.confounds import TSNR
import argparse


def setup(taskname, source_img, run_number):
    events = pd.read_csv(os.path.join(bidsdir, "task-" + taskname + "_events.tsv"), sep="\t")

    print('Using ' + source_img.filename + ' as source image.')
    confounds = pd.read_csv(os.path.join(fmriprepdir, "sub-" + source_img.entities['subject'],
                                         "ses-" + source_img.entities['session'], "func",
                                         "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                         'session'] + "_task-" + taskname + "_run-" + run_number +
                                         "_desc-confounds_regressors.tsv"), sep="\t", na_values="n/a")
    if aroma:
        print("AROMA config selected. Using ICA-AROMA denoised image.")
        subject_info = [Bunch(conditions=[taskname],
                              onsets=[list(events[events.trial_type == 'stimulus'].onset)],
                              durations=[list(events[events.trial_type == 'stimulus'].duration)])]

        prepped_img = os.path.join(fmriprepdir, "sub-" + source_img.entities['subject'],
                                   "ses-" + source_img.entities['session'], "func",
                                   "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                   'session'] + "_task-" + taskname + "_run-" + run_number +
                                   "_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz")
    else:
        subject_info = [Bunch(conditions=[taskname],
                              onsets=[list(events[events.trial_type == 'stimulus'].onset),
                                      list(events[events.trial_type == 'baseline'].onset)],
                              durations=[list(events[events.trial_type == 'stimulus'].duration),
                                         list(events[events.trial_type == 'baseline'].duration)],
                              regressors=[confounds['global_signal'],
                                          confounds['csf'],
                                          confounds['white_matter'],
                                          confounds['a_comp_cor_00'],
                                          confounds['a_comp_cor_01'],
                                          confounds['a_comp_cor_02'],
                                          confounds['a_comp_cor_03'],
                                          confounds['a_comp_cor_04'],
                                          confounds['a_comp_cor_05'],
                                          confounds['trans_x'],
                                          confounds['trans_y'],
                                          confounds['trans_z'],
                                          confounds['rot_x'],
                                          confounds['rot_y'],
                                          confounds['rot_z'],
                                          ],
                              regressor_names=['global_signal',
                                               'csf',
                                               'white_matter',
                                               'a_comp_cor_00',
                                               'a_comp_cor_01',
                                               'a_comp_cor_02',
                                               'a_comp_cor_03',
                                               'a_comp_cor_04',
                                               'a_comp_cor_05',
                                               'trans_x',
                                               'trans_y',
                                               'trans_z',
                                               'rot_x',
                                               'rot_y',
                                               'rot_z'
                                               ]
                              )]

        prepped_img = os.path.join(fmriprepdir, "sub-" + source_img.entities['subject'],
                                   "ses-" + source_img.entities['session'], "func",
                                   "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                   'session'] + "_task-" + taskname + "_run-" + run_number +
                                   "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

    print('Using ' + os.path.basename(prepped_img) + ' as preprocessed image.')

    return source_img, prepped_img, subject_info, confounds


def model_fitting(source_img, prepped_img, subject_info, task):
    taskdir = os.path.join(outputdir, task)
    if not os.path.exists(taskdir):
        os.mkdir(taskdir)

    # skull strip the preprocessed BOLD
    print("Skull-stripping the preprocessed BOLD.")
    bet = fsl.BET()
    bet.inputs.in_file = prepped_img
    bet.inputs.frac = 0.6
    bet.inputs.functional = True
    bet.inputs.out_file = os.path.join(taskdir, task + "_input_functional_bet.nii.gz")
    bet_res = bet.run()
    bettedinput = bet_res.outputs.out_file

    if aroma is True:
        print("No smoothing required.")
    else:
        # smoothing
        print("Smoothing the skull-stripped BOLD.")
        smooth = fsl.Smooth()
        smooth.inputs.in_file = bettedinput
        smooth.inputs.fwhm = 8
        smooth.inputs.smoothed_file = os.path.join(taskdir, task + "_input_functional_bet_smooth.nii.gz")
        smooth.run()
        smoothed_betted_input = os.path.join(taskdir, task + "_input_functional_bet_smooth.nii.gz")

    task_vs_baseline = [task + " vs baseline", 'T', [task, 'baseline'], [1, -1]]  # set up contrasts
    contrasts = [task_vs_baseline]

    """
    Inputs::
         inputspec.session_info : info generated by modelgen.SpecifyModel
         inputspec.interscan_interval : interscan interval
         inputspec.contrasts : list of contrasts
         inputspec.film_threshold : image threshold for FILM estimation
         inputspec.model_serial_correlations
         inputspec.bases
    Outputs::
         outputspec.copes
         outputspec.varcopes
         outputspec.dof_file
         outputspec.pfiles
         outputspec.zfiles
         outputspec.parameter_estimates
    """

    modelfit = pe.Workflow(name='modelfit', base_dir=taskdir)

    """
    Create nodes
    """

    modelspec = pe.Node(interface=model.SpecifyModel(), name="modelspec")  # generate design info
    inputspec = pe.Node(
        util.IdentityInterface(fields=[
            'session_info', 'interscan_interval', 'contrasts',
            'film_threshold', 'functional_data', 'bases',
            'model_serial_correlations'
        ]),
        name='inputspec')
    level1design = pe.Node(interface=fsl.Level1Design(), name="level1design")
    modelgen = pe.MapNode(
        interface=fsl.FEATModel(),
        name='modelgen',
        iterfield=['fsf_file', 'ev_files'])
    modelestimate = pe.MapNode(
        interface=fsl.FILMGLS(smooth_autocorr=True, mask_size=5),
        name='modelestimate',
        iterfield=['design_file', 'in_file', 'tcon_file'])
    merge_contrasts = pe.MapNode(
        interface=util.Merge(2), name='merge_contrasts', iterfield=['in1'])
    ztopval = pe.MapNode(
        interface=fsl.ImageMaths(op_string='-ztop', suffix='_pval'),
        nested=True,
        name='ztop',
        iterfield=['in_file'])
    outputspec = pe.Node(
        util.IdentityInterface(fields=[
            'copes', 'varcopes', 'dof_file', 'pfiles', 'zfiles',
            'parameter_estimates'
        ]),
        name='outputspec')

    modelfit.connect([
        (modelspec, inputspec, [('session_info', 'session_info')]),
        (inputspec, level1design,
         [('interscan_interval', 'interscan_interval'),
          ('session_info', 'session_info'), ('contrasts', 'contrasts'),
          ('bases', 'bases'), ('model_serial_correlations',
                               'model_serial_correlations')]),
        (inputspec, modelestimate, [('film_threshold', 'threshold'),
                                    ('functional_data', 'in_file')]),
        (level1design, modelgen, [('fsf_files', 'fsf_file'), ('ev_files',
                                                              'ev_files')]),
        (modelgen, modelestimate, [('design_file', 'design_file')]),
        (merge_contrasts, ztopval, [('out', 'in_file')]),
        (ztopval, outputspec, [('out_file', 'pfiles')]),
        (merge_contrasts, outputspec, [('out', 'zfiles')]),
        (modelestimate, outputspec, [('param_estimates',
                                      'parameter_estimates'), ('dof_file',
                                                               'dof_file')]),
    ])

    modelfit.connect([
        (modelgen, modelestimate, [('con_file', 'tcon_file'),
                                   ('fcon_file', 'fcon_file')]),
        (modelestimate, merge_contrasts, [('zstats', 'in1'), ('zfstats', 'in2')]),
        (modelestimate, outputspec, [('copes', 'copes'), ('varcopes', 'varcopes')]),
    ])

    # define inputs to workflow
    if aroma:
        modelspec.inputs.functional_runs = bettedinput
        inputspec.inputs.functional_data = bettedinput
    else:
        modelspec.inputs.functional_runs = smoothed_betted_input
        inputspec.inputs.functional_data = smoothed_betted_input
        print("Using smoothed image as input.")

    modelspec.inputs.subject_info = subject_info
    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.time_repetition = source_img.entities['RepetitionTime']
    modelspec.inputs.high_pass_filter_cutoff = 90
    inputspec.inputs.model_serial_correlations = True
    inputspec.inputs.film_threshold = 10.0
    inputspec.inputs.interscan_interval = source_img.entities['RepetitionTime']
    inputspec.inputs.bases = {'gamma': {'gammasigma': 3, 'gammadelay': 6, 'derivs': True}}
    inputspec.inputs.contrasts = contrasts

    # Run the model-fitting pipeline. Main outputs are a feat directory (w/ functional img) and a design.mat file
    res = modelfit.run()

    # outputs
    print('--------------------nodes------------------')
    print(list(res.nodes))
    for i in range(len(list(res.nodes)) - 1):
        print('---------------------------')
        print("%s: %s %s" % ("| NODE", list(res.nodes)[i], '|'))
        print('---------------------------')
        print('')
        print('----------outputs-----------')
        print(list(res.nodes)[i].result.outputs)
        print('')

    # the third node, FILM's, first element (i.e. only element) of its 'zstats' output
    z_img = list(res.nodes)[2].result.outputs.zstats[0]

    # estimate smoothness
    est = fsl.SmoothEstimate()
    est.inputs.zstat_file = z_img
    est.inputs.mask_file = os.path.join(taskdir, task + "_input_functional_bet_mask.nii.gz")
    est_run = est.run()
    dlh = est_run.outputs.dlh

    # cluster-wise post-stats thresholding
    cl = fsl.Cluster()
    cl.inputs.threshold = 3.0902
    cl.inputs.in_file = z_img
    cl.inputs.dlh = dlh
    cl.inputs.out_threshold_file = os.path.join(taskdir, task + '_cluster_thresholded_z.nii.gz')
    print('')
    print('CLUSTER OUTPUT')
    print('')
    print(cl.cmdline)
    cl_res = cl.run()
    print(cl.cmdline)
    print('threshold file: ' + cl_res.outputs.threshold_file)

    # resample the result image
    if aroma:
        thresh_img = cl_res.outputs.threshold_file
    else:
        print("Resampling thresholded image to MNI space")
        mni = '/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
        resampled_thresh_img = nilearn.image.resample_to_img(cl_res.outputs.threshold_file, mni)
        thresh_img = os.path.join(taskdir, task + '_cluster_thresholded_z_resample.nii.gz')
        resampled_thresh_img.to_filename(thresh_img)
        # threshold to remove artifacts from resampling
        thr = fsl.Threshold()
        thr.inputs.in_file = thresh_img
        thr.inputs.thresh = 0.001
        thr.inputs.out_file = os.path.join(taskdir, task + '_cluster_thresholded_z_resample_thr.nii.gz')
        thr.run()
        thresh_img = thr.inputs.out_file

    print("Image to be returned: " + thresh_img)

    return thresh_img


class PostStats:
    """
    Class to perform analyses on data
    """

    def __init__(self, img, task, rois, masks, confounds):
        self.img = img
        self.task = task
        self.taskdir = os.path.join(outputdir, self.task)
        self.rois = rois  # a list of strings used as labels in plot
        self.masks = masks  # masks to do statistics on
        self.confounds = confounds  # confounds tsv (returned from setup())

        self.left_stats, self.right_stats, self.ars = self.calc_stats()

    def create_glass_brain(self):
        if self.task == 'scenemem':
            masked_img_path = os.path.join(self.taskdir, self.task + "_img_masked.nii.gz")
            applymask = fsl.ApplyMask(in_file=self.img, mask_file=mtl_mask, out_file=masked_img_path)
            applymask.run()
            nilearn.plotting.plot_glass_brain(self.img,
                                              output_file=os.path.join(outputdir, self.task, self.task + "_gb.svg"),
                                              display_mode='lyrz', colorbar=True, plot_abs=False, threshold=1.5)
        else:
            nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(self.img, 8),
                                              output_file=os.path.join(outputdir, self.task, self.task + "_gb.svg"),
                                              display_mode='lyrz', colorbar=True, plot_abs=False, threshold=3.5)

        out_file = "./" + self.task + "/" + self.task + "_gb.svg"
        return out_file

    def get_mask_vox(self, msk):
        mask_stat = fsl.ImageStats(in_file=msk, op_string=' -V')
        mask_run = mask_stat.run()
        mask_vox = list(mask_run.outputs.out_stat)
        return mask_vox[0]

    def get_roi_perc(self, msk, mask_vox):
        roi_stat = fsl.ImageStats(in_file=self.img, op_string='-k ' + msk + ' -V')
        print(roi_stat.cmdline)
        stat_run = roi_stat.run()
        stat = float(list(stat_run.outputs.out_stat)[0])
        perc = (stat / mask_vox) * 100
        return perc

    def calc_ar(self, left, right):
        if not (left + right) > 0:
            return 0
        else:
            return (left - right) / (left + right)

    def calc_stats(self):
        masks = self.masks
        vox = {}  # dictionary of mask: mask voxels
        res = {}  # dictionary of roi: percent activation
        left_stats = []  # for plot
        right_stats = []
        ars = []  # list of asymmetry ratios for table
        for mask in masks:  # order is important -- it must correspond with the ROI labels (self.rois)
            roi = os.path.basename(mask).split('.')[0]
            vox[roi + '_vox'] = self.get_mask_vox(mask)
            res[roi] = round(self.get_roi_perc(mask, vox[roi + '_vox']))
            if "left" in roi:
                left_stats.append(res[roi])
            else:
                right_stats.append(res[roi])

        for i in range(0, len(left_stats)):
            ar_result = round(self.calc_ar(left_stats[i], right_stats[i]), 2)
            ars.append(ar_result)

        return left_stats, right_stats, ars

    def create_bar_plot(self):
        # Bar graph
        index = np.arange(len(self.left_stats))
        bar_width = 0.2
        opacity = 0.8
        axes = plt.gca()
        axes.set_ylim([0, 100])

        plt.bar(index, self.left_stats, bar_width,
                alpha=opacity,
                color='#4f6bb0',
                label='Left')
        plt.bar(index + bar_width, self.right_stats, bar_width,
                alpha=opacity,
                color='#550824',
                label='Right')

        plt.xlabel('ROI')
        plt.ylabel('% activated voxels in ROI')
        plt.title(self.task)
        plt.xticks(index + bar_width / 2, self.rois)
        plt.legend()
        plt.savefig(os.path.join(outputdir, self.task, self.task + "_bar.svg"))
        plt.close()

        plot_file = "./" + self.task + "/" + self.task + "_bar.svg"
        return plot_file

    def generate_statistics_table(self):
        row = self.rois
        column = ['left %', 'right %', 'asymmetry ratio']
        data = np.array([self.left_stats, self.right_stats, self.ars]).transpose()
        df = pd.DataFrame(data, index=row, columns=column)
        html_table = df.to_html()
        return html_table

    def calc_iqms(self):
        # tSNR
        tsnr = TSNR()
        tsnr.inputs.in_file = self.img
        tsnr.inputs.mean_file = os.path.join(outputdir, self.task, self.task + "_mean_tsnr.nii.gz")
        tsnr.inputs.stddev_file = os.path.join(outputdir, self.task, self.task + "stddev.nii.gz")
        tsnr_res = tsnr.run()
        mean_tsnr_img = tsnr_res.outputs.mean_file
        stat = fsl.ImageStats(in_file=mean_tsnr_img, op_string=' -M')
        stat_run = stat.run()
        mean_tsnr = stat_run.outputs.out_stat

        # framewise-displacement
        column_means = self.confounds.mean(axis=0, skipna=True)
        mean_fd = round(column_means['framewise_displacement'], 2)

        return mean_tsnr, mean_fd

    def create_html_viewer(self):
        html_view = nilearn.plotting.view_img(self.img, threshold=4, bg_img='MNI152', vmax=10,
                                              title=self.task)
        html_view.save_as_html(os.path.join(outputdir, self.task, self.task + "_viewer.html"))
        viewer_file = "./" + self.task + "/" + self.task + "_viewer.html"
        return viewer_file


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate a presurgical report from fmriprep results")
    parser.add_argument(
        "--bidsdir",
        help="Path to a curated BIDS directory",
        required=True
    )
    parser.add_argument(
        "--fmriprepdir",
        help="Path to fmriprep results directory",
        required=True
    )
    parser.add_argument(
        "--outputdir",
        help="Path to output directory",
        required=True
    )
    parser.add_argument(
        "--aroma",
        help="Use ICA-AROMA denoised BOLD images",
        action='store_true',
        default=True
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        help="Space separated list of tasks",
        required=True
    )

    return parser


def main():
    """
    Entry point for the script.
    Render a template and write it to file.
    :return:
    """

    # Content to be published
    title = "presurgical fMRI Report"

    # Produce our section blocks
    sections = list()

    # Add the first section, a summary list and legend
    sections.append(summary_section_template.render(
        subject_id=layout.get(return_type='id', target='subject')[0].strip("[']"),
        task_list=task_list,
        task_number=len(task_list),
        asym_ratio_eq=os.path.join(currdir, 'imgs', 'asym_ratio_equation.png')))

    # Add the navigation bar at the top
    sections.append(navbar_template.render(
        task_list=task_list
    ))

    # Do the analysis for each task. Each task has a unique set of ROIs
    for task in task_list:
        # get all the runs from the BIDS dataset, loop through if more than one
        run_list = layout.get(task=task, session="01", suffix="bold", extension="nii.gz")
        for i in range(0, len(run_list)):
            source_img = run_list[i]
            run_number = "0" + str(i + 1)
            (source_epi, input_functional, info, confounds) = setup(task, source_img, run_number)
            thresholded_img = model_fitting(source_epi, input_functional, info, task)
            if task == 'object':
                rois = ['whole brain', "broca's area"]
                masks = [lhem_mask, rhem_mask, lba_mask, rba_mask]
            elif task == 'rhyme':
                rois = ['whole brain', "broca's area", "wernicke's area"]
                masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lwa_mask, rwa_mask]
            elif task == 'scenemem':
                rois = ['mTL', 'hippocampus', 'fusiform gyrus', 'parahippocampal gyrus']
                masks = [lmtl_mask, rmtl_mask, lhc_mask, rhc_mask, lffg_mask, rffg_mask, lphg_mask, rphg_mask]
            elif task == 'sentence':
                rois = ['wb', "ba", "superior TG", "middle TG", "inferior TG"]
                masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lstg_mask, rstg_mask, lmtg_mask, rmtg_mask, litg_mask,
                         ritg_mask]
            elif task == 'wordgen':
                rois = ['whole brain', "broca's area", "superior frontal gyrus"]
                masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lsfg_mask, rsfg_mask]

            # create a PostStats object for the current task. Add elements to the section based on the object's methods
            post_stats = PostStats(thresholded_img, task, rois, masks, confounds)
            sections.append(task_section_template.render(
                section_name="ses-01_task-" + task + "_run-" + run_number,  # the link that IDs this section for the nav bar
                task_title=task,
                run_number=str(i + 1),
                len_run_list=len(run_list),
                mean_tsnr=post_stats.calc_iqms()[0],
                mean_fd=post_stats.calc_iqms()[1],
                gb_path=post_stats.create_glass_brain(),  # glass brain
                viewer_path=post_stats.create_html_viewer(),  # interactive statistical map viewer
                bar_path=post_stats.create_bar_plot(),  # bar plot
                table=post_stats.generate_statistics_table()  # statistics table
            ))

    # Produce and write the report to file
    with open(os.path.join(outputdir, "report.html"), "w") as f:
        f.write(base_template.render(
            title=title,
            sections=sections
        ))


fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
datadir = os.getcwd()
currdir = os.path.dirname(__file__)

# parse command line arguments
arg_parser = get_parser()
args = arg_parser.parse_args()

bidsdir = args.bidsdir
fmriprepdir = args.fmriprepdir
outputdir = args.outputdir
aroma = args.aroma
task_arg = args.tasks  # this returns a list with one item, a string with each task separated by a space
task_str = task_arg[0]  # take the string (first and only element of list)
task_list = task_str.split()  # and split it into a list with a string element for each task

# get the layout object of the BIDS directory
layout = BIDSLayout(bidsdir)

# define the masks
lba_mask = os.path.join(datadir, "masks", "ba_left.nii.gz")
rba_mask = os.path.join(datadir, "masks", "ba_right.nii.gz")
lstg_mask = os.path.join(datadir, "masks", "stg_left.nii.gz")
rstg_mask = os.path.join(datadir, "masks", "stg_right.nii.gz")
lmtg_mask = os.path.join(datadir, "masks", "mtg_left.nii.gz")
rmtg_mask = os.path.join(datadir, "masks", "mtg_right.nii.gz")
litg_mask = os.path.join(datadir, "masks", "itg_left.nii.gz")
ritg_mask = os.path.join(datadir, "masks", "itg_right.nii.gz")
lsfg_mask = os.path.join(datadir, "masks", "sfg_left.nii.gz")
rsfg_mask = os.path.join(datadir, "masks", "sfg_right.nii.gz")
lwa_mask = os.path.join(datadir, "masks", "stg_post_left.nii.gz")
rwa_mask = os.path.join(datadir, "masks", "stg_post_right.nii.gz")
lhem_mask = os.path.join(datadir, "masks", "hem_left.nii.gz")
rhem_mask = os.path.join(datadir, "masks", "hem_right.nii.gz")

mtl_mask = os.path.join(datadir, "masks", "hpf_bin.nii.gz")
lmtl_mask = os.path.join(datadir, "masks", "hpf_left_bin.nii.gz")
rmtl_mask = os.path.join(datadir, "masks", "hpf_right_bin.nii.gz")
lhc_mask = os.path.join(datadir, "masks", "hippocampus_left_bin.nii.gz")
rhc_mask = os.path.join(datadir, "masks", "hippocampus_right_bin.nii.gz")
lffg_mask = os.path.join(datadir, "masks", "ffg_left_bin.nii.gz")
rffg_mask = os.path.join(datadir, "masks", "ffg_right_bin.nii.gz")
lphg_mask = os.path.join(datadir, "masks", "phg_left_bin.nii.gz")
rphg_mask = os.path.join(datadir, "masks", "phg_right_bin.nii.gz")

# Configure Jinja and ready the templates
env = Environment(
    loader=FileSystemLoader(searchpath="templates")
)

# Assemble the templates we'll use
base_template = env.get_template("report.html")
summary_section_template = env.get_template("summary_section.html")
task_section_template = env.get_template("task_section.html")
navbar_template = env.get_template("navbar.html")

if __name__ == "__main__":
    main()
