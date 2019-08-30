#!/usr/bin/env python
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
import sys

import matplotlib.pyplot as plt
import nilearn.plotting
import nipype.algorithms.modelgen as model
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import numpy as np
import pandas as pd
from bids import BIDSLayout
from nipype.interfaces.base import Bunch
from jinja2 import FileSystemLoader, Environment


def setup(taskname):
    events = pd.read_csv(os.path.join(bidsdir, "task-" + taskname + "_events.tsv"), sep="\t")
    source_img = layout.get(task=taskname, run=1, session="01", suffix="bold", extension="nii.gz")[0]
    print('Using ' + source_img.filename + ' as source image.')

    subject_info = [Bunch(conditions=[taskname],
                          onsets=[list(events[events.trial_type == 'stimulus'].onset)],
                          durations=[list(events[events.trial_type == 'stimulus'].duration)])]

    prepped_img = os.path.join(fmriprepdir, "sub-" + source_img.entities['subject'],
                               "ses-" + source_img.entities['session'], "func",
                               "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                   'session'] + "_task-" + taskname + "_run-01_space-MNI152NLin2009cAsym_desc"
                                                                      "-smoothAROMAnonaggr_bold.nii.gz")
    return source_img, prepped_img, subject_info


def model_fitting(source_img, prepped_img, subject_info, task):
    taskdir = os.path.join(outputdir, task)
    if not os.path.exists(taskdir):
        os.mkdir(taskdir)

    # skull strip the preprocessed BOLD
    bet = fsl.BET()
    bet.inputs.in_file = prepped_img
    bet.inputs.frac = 0.7
    bet.inputs.functional = True
    bet.inputs.out_file = os.path.join(taskdir, task + "_input_functional_bet.nii.gz")
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


class PostStats:
    """
    Class to perform analyses on data
    """

    def __init__(self, img, task, rois, masks):
        self.img = img
        self.task = task
        self.taskdir = os.path.join(outputdir, self.task)
        self.rois = rois  # a list of strings used as labels in plot
        self.masks = masks  # masks to do statistics on

        self.left_stats, self.right_stats, self.ars = self.calc_stats()

    def create_glass_brain(self):
        if self.task == 'scenemem':
            # masked_img = fsl.ImageMaths(in_file=img, mask_file=mtl_mask, out_file=taskdir + task + "_img_masked.nii.gz")
            nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(self.img, 8),
                                              output_file=os.path.join(self.taskdir, self.task + "_gb.svg"),
                                              display_mode='lyrz', colorbar=True, plot_abs=False, threshold=5)
        else:
            nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(self.img, 8),
                                              output_file=os.path.join(self.taskdir, self.task + "_gb.svg"),
                                              display_mode='lyrz', colorbar=True, plot_abs=False, threshold=3.5)

        out_file = self.taskdir + self.task + "_gb.svg"
        return out_file

    def get_mask_vox(self, msk):
        mask_stat = fsl.ImageStats(in_file=msk, op_string=' -V')
        mask_run = mask_stat.run()
        mask_vox = list(mask_run.outputs.out_stat)
        return mask_vox[0]

    def get_roi_perc(self, img, msk, mask_vox):
        roi_stat = fsl.ImageStats(in_file=self.img, op_string='-k ' + msk + ' -V')
        print(roi_stat.cmdline)
        stat_run = roi_stat.run()
        stat = float(list(stat_run.outputs.out_stat)[0])
        perc = (stat / mask_vox) * 100
        return perc

    def calc_ar(self, left, right):
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
            res[roi] = round(self.get_roi_perc(self.img, mask, vox[roi + '_vox']))
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
        plt.savefig(os.path.join(self.taskdir, self.task + "_bar.svg"))
        plt.close()

        plot_file = os.path.join(self.taskdir, self.task + "_bar.svg")
        return plot_file

    def generate_statistics_table(self):
        row = self.rois
        column = ['left %', 'right %', 'asymmetry ratio']
        data = np.array([self.left_stats, self.right_stats, self.ars]).transpose()
        df = pd.DataFrame(data, index=row, columns=column)
        html_table = df.to_html()
        return html_table

    def create_html_viewer(self):
        html_view = nilearn.plotting.view_img(nilearn.image.smooth_img(self.img, 8), threshold=5, bg_img='MNI152', vmax=10,
                                              title=self.task)
        html_view.save_as_html(os.path.join(self.taskdir, self.task + "_viewer.html"))
        viewer_file = os.path.join(self.taskdir, self.task + "viewer.html")
        return viewer_file


fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
# datadir = '/home/will/PycharmProjects/report_gear/
datadir = os.getcwd()

try:
    bidsdir = sys.argv[1]
except IOError:
    print("No bids directory specified.")
    sys.exit(1)

try:
    fmriprepdir = sys.argv[2]
except IOError:
    print("No fmriprep directory specified.")
    sys.exit(1)

try:
    outputdir = sys.argv[3]
except IOError:
    print("No output directory specified.")
    sys.exit(1)


# get the layout object of the BIDS directory
layout = BIDSLayout(bidsdir)

lba_mask = datadir + "masks/ba_left.nii.gz"
rba_mask = datadir + "masks/ba_right.nii.gz"
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

# Configure Jinja and ready the templates
env = Environment(
    loader=FileSystemLoader(searchpath="templates")
)

# Assemble the templates we'll use
base_template = env.get_template("report.html")
summary_section_template = env.get_template("summary_section.html")
task_section_template = env.get_template("task_section.html")
navbar_template = env.get_template("navbar.html")


def main():
    """
    Entry point for the script.
    Render a template and write it to file.
    :return:
    """

    task_list = layout.get_tasks()
    if 'rest' in task_list:
        task_list.remove('rest')

    # Content to be published
    title = "presurgical fMRI Report"

    # Produce our section blocks
    sections = list()

    # Add the first section, a summary list and legend
    sections.append(summary_section_template.render(
        subject_id=layout.get(return_type='id', target='subject')[0].strip("[']"),
        task_list=task_list,
        task_number=len(task_list),
        tsnr_eq=datadir + 'imgs/tsnr_equation.png',
        mean_eq=datadir + 'imgs/mean_signal.png',
        std_dev_eq=datadir + 'imgs/std_dev.png',
        asym_ratio_eq=datadir + 'imgs/asym_ratio_equation.png'
    ))

    # Add the navigation bar at the top
    sections.append(navbar_template.render(
        task_list=task_list
    ))

    # Do the analysis for each task. Each task has a unique set of ROIs
    for task in task_list:
        (source_epi, input_functional, info) = setup(task)
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
            masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lstg_mask, rstg_mask, lmtg_mask, rmtg_mask, litg_mask, ritg_mask]
        elif task == 'object':
            rois = ['whole brain', "broca's area", "superior frontal gyrus"]
            masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lsfg_mask, rsfg_mask]

        # create a PostStats object for the current task. Add elements to the section based on the object's methods
        post_stats = PostStats(thresholded_img, task, rois, masks)
        sections.append(task_section_template.render(
            section_name="ses-01_task-" + task + "_run-01",  # the link that IDs this section for the nav bar
            task_title=task,
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


if __name__ == "__main__":
    main()
