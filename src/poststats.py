#
# Class for computing statistics related to language lateralization and for generating graphs and figures representing these
# statistics, which will be bundled into the final HTML report
#
#
#

import matplotlib.pyplot as plt
import numpy as np
from nipype.algorithms.confounds import TSNR
import os
import nipype.interfaces.fsl as fsl
import nilearn.plotting
import pandas as pd


class PostStats:
    """
    Class to perform analyses on data
    """

    def __init__(self, subject_id, source_img, img, task, rois, masks, all_rois, all_masks, confounds, outputdir, datadir):
        self.subject_id = subject_id
        self.source_img = source_img.path
        self.img = img
        self.task = task
        self.outputdir = outputdir
        self.taskdir = os.path.join(outputdir, self.task)
        self.rois = rois  # a list of strings used as labels in plot
        self.masks = masks  # masks to do statistics on
        self.all_rois = all_rois
        self.all_masks = all_masks
        self.confounds = confounds  # confounds tsv (returned from setup())
        self.mtl_mask = os.path.join(datadir, "masks", "hpf_bin.nii.gz")

        self.vox_left_stats, self.vox_right_stats,  self.leftns, self.rightns, self.vox_ars, = self.calc_stats('vox', self.masks)
        self.mean_left_stats, self.mean_right_stats, self.leftns, self.rightns, self.mean_ars = self.calc_stats('mean', self.masks)
        self.all_vox_left_stats, self.all_vox_right_stats, self.all_leftns, self.all_rightns, self.all_vox_ars = self.calc_stats('vox', self.all_masks)

    def create_glass_brain(self):
        nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(self.img, 4),
                                          output_file=os.path.join(self.outputdir, self.task, 'figs', self.task + "_gb.svg"),
                                          display_mode='lyrz', colorbar=True, plot_abs=False, threshold=0)

        out_svg = self.task + "/figs/" + self.task + "_gb.svg"
        return out_svg

    def get_mask_vox(self, msk):
        mask_stat = fsl.ImageStats(in_file=msk, op_string=' -V')
        mask_run = mask_stat.run()
        mask_vox = list(mask_run.outputs.out_stat)
        return mask_vox[0]

    def get_roi_perc(self, msk, mask_vox):
        roi_stat = fsl.ImageStats(in_file=self.img, op_string='-k ' + msk + ' -l 0 -V')
        print(roi_stat.cmdline)
        stat_run = roi_stat.run()
        stat = float(list(stat_run.outputs.out_stat)[0])
        perc = (stat / mask_vox) * 100
        return perc

    def get_roi_activated_vox(self, msk):
        roi_stat = fsl.ImageStats(in_file=self.img, op_string='-k ' + msk + ' -l 0 -V')
        print(roi_stat.cmdline)
        stat_run = roi_stat.run()
        stat = float(list(stat_run.outputs.out_stat)[0])
        return stat

    def get_roi_mean_stat(self, msk):
        mean_zstat = fsl.ImageStats(in_file=self.img, op_string='-k ' + msk + ' -m')
        print(mean_zstat.cmdline)
        stat_run = mean_zstat.run()
        zscore = float(stat_run.outputs.out_stat)
        return zscore

    def calc_ar(self, left, right):
        if not (left + right) > 0:
            return 0
        else:
            return (left - right) / (left + right)

    def calc_stats(self, mode, masks):
        """
        Calculate activation statistics in the current list of ROIs
        :param mode: do percent activated voxels in  ROI (vox) or the mean z-score in ROI (mean)
        :return: left_stats: list of statistics in the left hemisphere ROIs
                 right_stats: list of statistics in the right hemisphere ROIs
                 ars: list of asymmetry ratios for each ROI
        """
        masks = masks
        vox = {}  # dictionary of mask: mask voxels
        res = {}  # dictionary of roi: percent activation
        n = {} # dictionary of n activated voxels in ROI
        mean = {}  # dictionary of mean z-score in ROI
        left_stats = []  # for plot
        right_stats = []
        leftns = []
        rightns = []
        ars = []  # list of asymmetry ratios for table
        for mask in masks:  # order is important -- it must correspond with the ROI labels (self.rois)
            roi = os.path.basename(mask).split('.')[0]
            if mode == 'vox':
                vox[roi + '_vox'] = self.get_mask_vox(mask)
                res[roi] = round(self.get_roi_perc(mask, vox[roi + '_vox']))
                number = res[roi]

            elif mode == 'mean':
                mean[roi] = round(self.get_roi_mean_stat(mask), 2)
                number = mean[roi]
            else:
                number = -999

            n[roi] = round(self.get_roi_activated_vox(mask))
            number_of_voxels = n[roi]

            if "left" in roi:
                left_stats.append(number)
                leftns.append(number_of_voxels)
            else:
                right_stats.append(number)
                rightns.append(number_of_voxels)

        for i in range(0, len(left_stats)):
            ar_result = round(self.calc_ar(left_stats[i], right_stats[i]), 2)
            ars.append(ar_result)

        return left_stats, right_stats, leftns, rightns, ars

    def create_bar_plot(self, left_stats, right_stats, mode):
        # Bar graph
        index = np.arange(len(left_stats))
        bar_width = 0.2
        opacity = 0.8
        axes = plt.gca()
        if mode == 'vox':
            axes.set_ylim([0, 100])
        else:
            axes.set_ylim([0,6])

        plt.bar(index, left_stats, bar_width,
                alpha=opacity,
                color='#4f6bb0',
                label='Left')
        plt.bar(index + bar_width, right_stats, bar_width,
                alpha=opacity,
                color='#550824',
                label='Right')

        plt.xlabel('ROI')
        if mode == "vox":
            plt.ylabel('% activated voxels in ROI')
        elif mode == "mean":
            plt.ylabel('mean z-score in ROI')
        plt.title(self.task)
        plt.xticks(index + bar_width / 2, self.rois)
        plt.legend()
        plt.savefig(os.path.join(self.outputdir, self.task, 'figs', self.task + "_" + mode + "_bar.svg"))
        plt.close()

        plot_file = "./" + self.task + "/figs/" + self.task + "_" + mode + "_bar.svg"
        return plot_file

    def create_vox_bar_plot(self):
        return self.create_bar_plot(self.vox_left_stats, self.vox_right_stats, 'vox')

    def create_mean_bar_plot(self):
        return self.create_bar_plot(self.mean_left_stats, self.mean_right_stats, 'mean')

    def generate_statistics_table(self, left_stats, right_stats, ars, mode):
        row = self.rois
        # table for HTML report
        columns = ['left', 'right', 'LI']
        if mode == 'vox':
            columns = ['left %', 'right %', 'LI']
        elif mode == 'mean':
            columns = ['left', 'right', 'LI']
        data = np.array([left_stats, right_stats, ars]).transpose()
        df = pd.DataFrame(data, index=row, columns=columns)
        df.to_csv(os.path.join(self.outputdir, self.task, 'figs', self.task + "_" + mode + "_html_table.csv"))
        html_table = df.to_html()

        return html_table

    def generate_csv(self, left_stats, right_stats, leftns, rightns, ars, rois):
        # table for csv output
        tSNR = self.calc_iqms()[0]
        FD = self.calc_iqms()[1]

        # Create lists of labels for each region: e.g. hemisphere left %, broca's left %, hemisphere right%, broca's right % etc.
        column_labels = []
        left_per = []
        right_per = []
        left_n = []
        right_n = []
        li_labels = []
        sum_labels = []
        for region in rois:
            left_per.append(region + ' left %')
            right_per.append(region + ' right %')
            left_n.append(region + ' left voxels')
            right_n.append(region + ' right voxels')
            li_labels.append(region + ' LI')
            sum_labels.append(region + ' sum')
        column_labels = ['subject'] + ['task'] + left_per + right_per + li_labels + left_n + right_n + sum_labels + ['FramewiseDisplacement'] + ['tSNR']
        sums = [sum(x) for x in zip(leftns, rightns)]
        row = [self.subject_id] + [self.task] + left_stats + right_stats + ars + leftns + rightns + sums + [FD] + [tSNR]
        data = np.array([row])
        df = pd.DataFrame(data, columns=column_labels)
        df.set_index('subject', inplace=True)
        df.to_csv(os.path.join(self.outputdir, self.task, 'stats', self.task + "_data.csv"))

    def generate_csv_wrap(self, task):
        if task == 'scenemem':
            self.generate_csv(self.vox_left_stats, self.vox_right_stats, self.leftns, self.rightns, self.vox_ars, self.rois)
        else:
            self.generate_csv(self.all_vox_left_stats, self.all_vox_right_stats, self.all_leftns, self.all_rightns, self.all_vox_ars, self.all_rois)

    def generate_vox_statistics_table(self):
        return self.generate_statistics_table(self.vox_left_stats, self.vox_right_stats, self.vox_ars, 'vox')

    def generate_mean_statistics_table(self):
        return self.generate_statistics_table(self.mean_left_stats, self.mean_right_stats, self.mean_ars, 'mean')

    def calc_iqms(self):
        # tSNR
        tsnr = TSNR()
        tsnr.inputs.in_file = self.source_img
        tsnr.inputs.mean_file = os.path.join(self.outputdir, self.task, self.task + "_mean_tsnr.nii.gz")
        tsnr_res = tsnr.run()
        mean_tsnr_img = tsnr_res.outputs.mean_file
        stat = fsl.ImageStats(in_file=mean_tsnr_img, op_string=' -M')
        stat_run = stat.run()
        mean_tsnr = round(stat_run.outputs.out_stat, 2)
        # framewise-displacement
        if type(self.confounds) == str:  # ensure self.confounds doesn't refer to empty string
            mean_fd = 'n/a'
        else:
            column_means = self.confounds.mean(axis=0, skipna=True)
            mean_fd = round(column_means['framewise_displacement'], 2)

        return mean_tsnr, mean_fd

    def create_html_viewer(self):
        mi = fsl.MeanImage()
        mi_run = mi.run(in_file=os.path.join(self.taskdir, self.task + "_input_functional_masked.nii.gz"))
        mean_img_path = mi_run.outputs.out_file
        html_view = nilearn.plotting.view_img(self.img, threshold=0, bg_img=mean_img_path, vmax=10,
                                              title=self.task)
        html_view.save_as_html(os.path.join(self.outputdir, self.task, 'figs', self.task + "_viewer.html"))
        viewer_file = "./" + self.task + "/figs/" + self.task + "_viewer.html"
        return viewer_file
