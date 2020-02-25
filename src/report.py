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
import pandas as pd
import poststats
import argparse
import modelfit
import os
import nipype.interfaces.fsl as fsl
from bids import BIDSLayout
from nipype.interfaces.base import Bunch
from jinja2 import FileSystemLoader, Environment
from weasyprint import HTML, CSS


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
    if taskname == 'scenemem' and os.path.isfile(aroma_path):
        aroma = True

    # Check if AROMA-denoised images exist. If they don't and the aroma config is selected, that's bad news
    if aroma and not os.path.isfile(aroma_path):
        print("You selected the AROMA configuration, but no AROMA-denoised images were found."
              " Using standard preprocessed bold images.")
        aroma = False

    # Check if confounds files actually exist and what to do if they don't
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

    if aroma and os.path.isfile(aroma_path):  # if AROMA config selected and the images exist
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

    mask_file = os.path.join(fmriprepdir, "sub-" + source_img.entities['subject'],
                              "ses-" + source_img.entities['session'], "func",
                              "sub-" + source_img.entities['subject'] + "_ses-" + source_img.entities[
                                  'session'] + "_task-" + taskname + "_run-" + run_number +
                              "_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")

    print('Using ' + os.path.basename(prepped_img) + ' as preprocessed image.')

    return source_img, prepped_img, subject_info, confounds, mask_file


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
    parser.add_argument(
        "--alpha",
        help="alpha value for thresholding",
        type=float
    )

    return parser


def generate_report():
    """
    Run the model-fitting and statistics pipelines.
    Render a report and write it to file.
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
        alpha=alpha,
        task_list=task_list,
        task_number=len(task_list),
        asym_ratio_eq='imgs/asym_ratio_equation.png'))

    # Add the navigation bar at the top
    sections.append(navbar_template.render(
        task_list=task_list
    ))

    # Branch off the section
    pdf_sections = list(sections)

    # Do the analysis for each task. Each task has a unique set of ROIs
    for task in task_list:
        # get all the runs from the BIDS dataset, loop through if more than one
        run_list = layout.get(task=task, suffix="bold", extension="nii.gz")
        for i in range(0, len(run_list)):
            source_img = run_list[i]
            run_number = "0" + str(i + 1)
            try:
                (source_epi, input_functional, info, confounds, mask_file) = setup(task, source_img, run_number)
            except FileNotFoundError:
                continue

            thresholded_img = modelfit.model_fitting(source_epi, input_functional, info, aroma, task, args, mask_file, i)

            def append_task_section(sec_list, is_png):
                all_rois = ["whole brain", "broca's area", "inferior frontal gyrus", "middle frontal gyrus",
                            "superior frontal gyrus", "frontal lobe", "inferior temporal gyrus", "middle temporal gyrus",
                            "superior temporal gyrus", "planum temporale", "angular gyrus"]
                all_masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lifg_mask, rifg_mask, lmfg_mask, rmfg_mask, lsfg_mask,
                             rsfg_mask, lfront_mask, rfront_mask, litg_mask, ritg_mask, lmtg_mask, rmtg_mask, lstg_mask, rstg_mask,
                             lpt_mask, rpt_mask, lag_mask, rag_mask]
                rois = []
                masks = []
                if task == 'object':
                    rois = ['whole brain', "broca's area", "inf. frontal", "mid. frontal", "planum temporale", "angular gyrus"]
                    masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lifg_mask, rifg_mask, lmfg_mask, rmfg_mask, lpt_mask,
                             rpt_mask, lag_mask, rag_mask]
                elif task == 'rhyme':
                    rois = ['whole brain', "broca's area", "frontal lobe", "planum temporale", "angular gyrus"]
                    masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lfront_mask, rfront_mask, lpt_mask, rpt_mask, lag_mask,
                             rag_mask]
                elif task == 'scenemem':
                    if is_png:
                        rois = ['mTL', 'hippocampus', 'amygdala', 'phg', 'entorhinal']
                    else:
                        rois = ['mTL', 'hippocampus', 'amygdala', 'parahippocampal gyrus', 'entorhinal cortex']
                    masks = [lmtl_mask, rmtl_mask, lhc_mask, rhc_mask, lam_mask, ram_mask, lphg_mask, rphg_mask, lent_mask,
                             rent_mask]
                elif task == 'sentence':
                    rois = ['wb', "ba", "sup. TG", "mid. TG", "inf. TG", "pt", "ag"]
                    masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lstg_mask, rstg_mask, lmtg_mask, rmtg_mask, litg_mask,
                             ritg_mask, lpt_mask, rpt_mask, lag_mask, rag_mask]
                elif task == 'wordgen':
                    if is_png:
                        rois = ['whole brain', "broca's area", "sfg", "ifg", "front", "pt", "ag"]
                    else:
                        rois = ['whole brain', "broca's area", "superior frontal gyrus", "inferior frontal gyrus", "frontal lobe",
                                "planum temporale", "angular gyrus"]
                    masks = [lhem_mask, rhem_mask, lba_mask, rba_mask, lsfg_mask, rsfg_mask, lifg_mask, rifg_mask, lfront_mask,
                             rfront_mask,
                             lpt_mask, rpt_mask, lag_mask, rag_mask]

                # create a PostStats object for the current task. Add elements to the section based on the object's methods
                post_stats = poststats.PostStats(sid, source_img, thresholded_img, task, rois, masks, all_rois, all_masks,
                                                 confounds, outputdir, datadir)
                sec_list.append(task_section_template.render(
                    section_name="task-" + task + "_run-" + run_number,  # the link that IDs this section for the nav bar
                    task_title=task,
                    is_png=is_png,
                    run_number=str(i + 1),
                    len_run_list=len(run_list),
                    mean_tsnr=post_stats.calc_iqms()[0],
                    mean_fd=post_stats.calc_iqms()[1],
                    gb_path=post_stats.create_glass_brain(),  # glass brain
                    viewer_path=post_stats.create_html_viewer(),  # interactive statistical map viewer
                    bar_path=post_stats.create_bar_plot(),  # bar plots
                    table=post_stats.generate_statistics_table(),  # statistics tables
                ))
                post_stats.generate_csv_wrap(task)

            append_task_section(sections, False)
            append_task_section(pdf_sections, True)

    # Produce and write the report to file
    with open(os.path.join(outputdir, "sub-" + sid + "_report.html"), "w") as f:
        f.write(base_template.render(
            title=title,
            sections=sections
        ))
    with open(os.path.join(outputdir, "report_png.html"), "w") as f:
        f.write(base_template.render(
            title=title,
            sections=pdf_sections
        ))
    html = HTML(os.path.join(outputdir, "report_png.html"))
    css = CSS(string='@page { size: A0 landscape; margin: .25cm }')
    html.write_pdf(
        os.path.join(outputdir, "sub-" + sid + "_report.pdf"), stylesheets=[css])
    os.remove(os.path.join(outputdir, "report_png.html"))


if __name__ == "__main__":
    datadir = os.getcwd()
    currdir = os.path.dirname(__file__)

    # define the masks
    lhem_mask = os.path.join(datadir, "masks", "hem_left.nii.gz")
    rhem_mask = os.path.join(datadir, "masks", "hem_right.nii.gz")
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
    lpt_mask = os.path.join(datadir, "masks", "pt_left.nii.gz")
    lmfg_mask = os.path.join(datadir, "masks", "mfg_left.nii.gz")
    rmfg_mask = os.path.join(datadir, "masks", "mfg_right.nii.gz")
    lifg_mask = os.path.join(datadir, "masks", "ifg_left.nii.gz")
    rifg_mask = os.path.join(datadir, "masks", "ifg_right.nii.gz")
    rpt_mask = os.path.join(datadir, "masks", "pt_right.nii.gz")
    lag_mask = os.path.join(datadir, "masks", "ang_left.nii.gz")
    rag_mask = os.path.join(datadir, "masks", "ang_right.nii.gz")
    rfront_mask = os.path.join(datadir, "masks", "frontal_right.nii.gz")
    lfront_mask = os.path.join(datadir, "masks", "frontal_left.nii.gz")

    mtl_mask = os.path.join(datadir, "masks", "mTL.nii.gz")
    lmtl_mask = os.path.join(datadir, "masks", "mTL_left.nii.gz")
    rmtl_mask = os.path.join(datadir, "masks", "mTL_right.nii.gz")
    lhc_mask = os.path.join(datadir, "masks", "hippocampus_left.nii.gz")
    rhc_mask = os.path.join(datadir, "masks", "hippocampus_right.nii.gz")
    lam_mask = os.path.join(datadir, "masks", "amygdala_left.nii.gz")
    ram_mask = os.path.join(datadir, "masks", "amygdala_right.nii.gz")
    lphg_mask = os.path.join(datadir, "masks", "phg_left.nii.gz")
    rphg_mask = os.path.join(datadir, "masks", "phg_right.nii.gz")
    lent_mask = os.path.join(datadir, "masks", "ento_left.nii.gz")
    rent_mask = os.path.join(datadir, "masks", "ento_right.nii.gz")

    template = os.path.join(datadir, "masks", "mni152.nii.gz")

    # Parse command line arguments
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
    alpha = args.alpha

    # Get the layout object of the BIDS directory
    layout = BIDSLayout(bidsdir)

    # Configure Jinja and ready the templates
    env = Environment(
        loader=FileSystemLoader(searchpath="templates")
    )

    # Assemble the templates we'll use
    base_template = env.get_template("report.html")
    summary_section_template = env.get_template("summary_section.html")
    task_section_template = env.get_template("task_section.html")
    navbar_template = env.get_template("navbar.html")

    generate_report()
