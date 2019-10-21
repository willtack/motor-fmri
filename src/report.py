#
#   This is the main script of the presurgical report gear. It performs model-fitting of fMRI runs preprocessed by fMRIPREP
#   to generate activation maps, computes statistics related to language lateralization with a PostStats object and compiles
#   these results into an html report. The program is intended to analyze fMRI scans of patients undergoing presurgical evaluation.
#
#   Will Tackett, University of Pennsylvania
#
#   August 13th, 2019
#

from __future__ import division
from __future__ import print_function

import os

import nilearn.plotting
import nipype.algorithms.modelgen as model
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import pandas as pd
from bids import BIDSLayout
import nistats
from nistats import thresholding
from nipype.interfaces.base import Bunch
from jinja2 import FileSystemLoader, Environment
from weasyprint import HTML, CSS
# from src import poststats
import poststats
import argparse
import nibabel


def setup(taskname, source_img, run_number):
    global aroma
    events = pd.read_csv(os.path.join(bidsdir, "task-" + taskname + "_events.tsv"), sep="\t")

    print('Using ' + source_img.filename + ' as source image.')
    confounds_path = os.path.join(fmriprepdir, "sub-" + source_img.entities['subject'],
                                         "ses-" + source_img.entities['session'], "func",
                                         "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                         'session'] + "_task-" + taskname + "_run-" + run_number +
                                         "_desc-confounds_regressors.tsv")
    aroma_path = os.path.join(fmriprepdir, "sub-" + source_img.entities['subject'],
                                   "ses-" + source_img.entities['session'], "func",
                                   "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                   'session'] + "_task-" + taskname + "_run-" + run_number +
                                   "_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz")

    simple_design = False
    confounds = ''

    # Always use AROMA-denoised images for the scenemem task (if they're there)
    if taskname=='scenemem' and os.path.isfile(aroma_path):
        aroma = True

    if os.path.isfile(confounds_path):
        print("Found confounds tsv at %s" % confounds_path)
        confounds = pd.read_csv(confounds_path, sep="\t", na_values="n/a")
    elif not os.path.isfile(confounds_path) and os.path.isfile(aroma_path):
        print("Confounds file not found...using AROMA-denoised image")
        aroma = True
    else:
        print("Could not find confounds file or AROMA-denoised image."
              " Using simplest design matrix. WARNING: resulting maps will be noisy.")
        simple_design = True

    if aroma and os.path.isfile(aroma_path): # if AROMA config selected and the images exist
        print("AROMA config selected. Using ICA-AROMA denoised image.")
        subject_info = [Bunch(conditions=[taskname],
                              onsets=[list(events[events.trial_type == 'stimulus'].onset),
                                      list(events[events.trial_type == 'baseline'].onset)],
                              durations=[list(events[events.trial_type == 'stimulus'].duration),
                                         list(events[events.trial_type == 'baseline'].duration)])]

        prepped_img = os.path.join(fmriprepdir, "sub-" + source_img.entities['subject'],
                                   "ses-" + source_img.entities['session'], "func",
                                   "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                   'session'] + "_task-" + taskname + "_run-" + run_number +
                                   "_space-MNI152NLin2009cAsym_desc-smoothAROMAnonaggr_bold.nii.gz")
    else:
        if simple_design:
            subject_info = [Bunch(conditions=[taskname],
                                  onsets=[list(events[events.trial_type == 'stimulus'].onset),
                                          list(events[events.trial_type == 'baseline'].onset)],
                                  durations=[list(events[events.trial_type == 'stimulus'].duration),
                                             list(events[events.trial_type == 'baseline'].duration)])]
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
                                                   ])]

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
    bet.inputs.mask = True
    bet.inputs.out_file = os.path.join(taskdir, task + "_input_functional_bet.nii.gz")
    bet_res = bet.run()
    betted_input = bet_res.outputs.out_file

    if aroma is True:
        print("No smoothing required.")
    else:
        # smoothing
        print("Smoothing the skull-stripped BOLD.")
        smooth = fsl.Smooth()
        smooth.inputs.in_file = betted_input
        smooth.inputs.fwhm = fwhm
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
          ('bases', 'bases'), ('model_serial_correlations', 'model_serial_correlations')]),
        (inputspec, modelestimate, [('film_threshold', 'threshold'),
                                    ('functional_data', 'in_file')]),
        (level1design, modelgen, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
        (modelgen, modelestimate, [('design_file', 'design_file')]),
        (merge_contrasts, ztopval, [('out', 'in_file')]),
        (ztopval, outputspec, [('out_file', 'pfiles')]),
        (merge_contrasts, outputspec, [('out', 'zfiles')]),
        (modelestimate, outputspec, [('param_estimates', 'parameter_estimates'), ('dof_file', 'dof_file')]),
    ])

    modelfit.connect([
        (modelgen, modelestimate, [('con_file', 'tcon_file'), ('fcon_file', 'fcon_file')]),
        (modelestimate, merge_contrasts, [('zstats', 'in1'), ('zfstats', 'in2')]),
        (modelestimate, outputspec, [('copes', 'copes'), ('varcopes', 'varcopes')]),
    ])

    # define inputs to workflow
    if aroma:
        modelspec.inputs.functional_runs = betted_input
        inputspec.inputs.functional_data = betted_input
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
    f = open(os.path.join(taskdir, task + '_outputs.txt'), 'w')
    print('', file=f)
    print(list(res.nodes), file=f)
    print('', file=f)
    for i in range(len(list(res.nodes)) - 1):
        print("%s: %s" % ("NODE", list(res.nodes)[i]), file=f)
        print(list(res.nodes)[i].result.outputs, file=f)
        print('', file=f)
    f.close()

    # the third node, FILM's, first element (i.e. only element) of its 'zstats' output
    z_img = list(res.nodes)[2].result.outputs.zstats[0]

    # Use False Discovery Rate theory to correct for multiple comparisons
    fdr_thresh_img, fdr_threshold = nistats.thresholding.map_threshold(stat_img=z_img,
                                                        mask_img=os.path.join(taskdir, task + "_input_functional_bet_mask.nii.gz"),
                                                        level=0.05,
                                                        height_control='fdr',
                                                        cluster_threshold=cthresh)
    print("Thresholding at FDR corrected threshold of " + str(fdr_threshold))
    fdr_thresh_img_path = os.path.join(taskdir, task + '_fdr_thresholded_z.nii.gz')
    nibabel.save(fdr_thresh_img, fdr_thresh_img_path)

    # Do a cluster analysis using the FDR corrected threshold on the original z_img
    cl = fsl.Cluster(in_file=z_img, threshold=fdr_threshold)
    cl_run = cl.run()
    clusters = cl_run.runtime.stdout  # write the terminal output to a text file
    cluster_file = os.path.join(taskdir, task + "_cluster_stats.txt")
    f = open(cluster_file, 'w')
    f.write(clusters)
    f.close()
    # resample the result image
    if aroma:
        thresh_img = fdr_thresh_img_path
    else:
        print("Resampling thresholded image to MNI space")
        resampled_thresh_img = nilearn.image.resample_to_img(fdr_thresh_img_path, template)
        thresh_img = os.path.join(taskdir, task + '_fdr_thresholded_z_resample_1.nii.gz')
        resampled_thresh_img.to_filename(thresh_img)
        # threshold to remove artifacts from resampling
        thr = fsl.Threshold()
        thr.inputs.in_file = thresh_img
        thr.inputs.thresh = 0.001
        thr.inputs.out_file = os.path.join(taskdir, task + '_fdr_thresholded_z_resample.nii.gz')
        thr.run()
        thresh_img = thr.inputs.out_file

    print("Image to be returned: " + thresh_img)

    return thresh_img


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
        default=False
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        help="Space separated list of tasks",
        required=True
    )
    parser.add_argument(
        "--fwhm",
        help="size of smoothing kernel",
        type=int
    )
    parser.add_argument(
        "--cthresh",
        help="Cluster extent threshold",
        type=int
    )

    return parser


def generate_report(n):
    """
    Inputs::
    Render a template and write it to file.
    """

    # Content to be published
    title = "presurgical fMRI Report"

    # Produce our section blocks
    sections = list()

    # Subject ID
    sid = layout.get(return_type='id', target='subject')[0].strip("[']")

    # Add the first section, a summary list and legend
    sections.append(summary_section_template.render(
        subject_id=sid,
        fwhm=fwhm,
        cthresh=cthresh,
        task_list=task_list,
        task_number=len(task_list),
        asym_ratio_eq='imgs/asym_ratio_equation.png'))

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
            try:
                (source_epi, input_functional, info, confounds) = setup(task, source_img, run_number)
            except FileNotFoundError:
                continue

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
            post_stats = poststats.PostStats(thresholded_img, task, rois, masks, confounds, outputdir, datadir)
            sections.append(task_section_template.render(
                section_name="ses-01_task-" + task + "_run-" + run_number,  # the link that IDs this section for the nav bar
                task_title=task,
                run_number=str(i + 1),
                len_run_list=len(run_list),
                mean_tsnr=post_stats.calc_iqms()[0],
                mean_fd=post_stats.calc_iqms()[1],
                gb_path=post_stats.create_glass_brain(),  # glass brain
                viewer_path=post_stats.create_html_viewer(),  # interactive statistical map viewer
                vox_bar_path=post_stats.create_vox_bar_plot(),  # bar plots
                mean_bar_path=post_stats.create_mean_bar_plot(),
                vox_table=post_stats.generate_vox_statistics_table(),  # statistics tables
                mean_table=post_stats.generate_mean_statistics_table()
            ))

    # Produce and write the report to file
    with open(os.path.join(outputdir, "report.html"), "w") as f:
        f.write(base_template.render(
            title=title,
            sections=sections
        ))
    with open(os.path.join(outputdir, "report_png.html"), "w") as f:
        f.write(base_template.render(
            title=title,
            sections=sections
        ))
    html = HTML(os.path.join(outputdir, "report_png.html"))
    css = CSS(string='@page { size: A0 landscape; margin: .25cm }')
    html.write_pdf(
        os.path.join(outputdir, "sub-" + sid + "_report.pdf"), stylesheets=[css])


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
fwhm = args.fwhm
cthresh = args.cthresh

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
template = os.path.join(datadir, "masks", "mni152.nii.gz")

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
    generate_report()
