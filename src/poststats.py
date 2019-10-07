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
import nilearn
import pandas as pd


class PostStats:
    """
    Class to perform analyses on data
    """

    def __init__(self, img, task, rois, masks, confounds, outputdir, datadir):
        self.img = img
        self.task = task
        self.outputdir = outputdir
        self.taskdir = os.path.join(outputdir, self.task)
        self.rois = rois  # a list of strings used as labels in plot
        self.masks = masks  # masks to do statistics on
        self.confounds = confounds  # confounds tsv (returned from setup())
        self.mtl_mask = os.path.join(datadir, "masks", "hpf_bin.nii.gz")

        self.left_stats, self.right_stats, self.ars = self.calc_stats()

    def create_glass_brain(self):
        if self.task == 'scenemem':
            masked_img_path = os.path.join(self.taskdir, self.task + "_img_masked.nii.gz")
            applymask = fsl.ApplyMask(in_file=self.img, mask_file=self.mtl_mask, out_file=masked_img_path)
            applymask.run()
            nilearn.plotting.plot_glass_brain(masked_img_path,
                                              output_file=os.path.join(self.outputdir, self.task, self.task + "_gb.svg"),
                                              display_mode='lyrz', colorbar=True, plot_abs=False, threshold=1.5)
        else:
            nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(self.img, 4),
                                              output_file=os.path.join(self.outputdir, self.task, self.task + "_gb.svg"),
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
        plt.savefig(os.path.join(self.outputdir, self.task, self.task + "_bar.svg"))
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
        tsnr.inputs.mean_file = os.path.join(self.outputdir, self.task, self.task + "_mean_tsnr.nii.gz")
        tsnr_res = tsnr.run()
        mean_tsnr_img = tsnr_res.outputs.mean_file
        stat = fsl.ImageStats(in_file=mean_tsnr_img, op_string=' -M')
        stat_run = stat.run()
        mean_tsnr = stat_run.outputs.out_stat
        # framewise-displacement
        if type(self.confounds) == str:  # test that confounds is the pd tsv, not the empty string
            mean_fd = 'n/a'
        else:
            column_means = self.confounds.mean(axis=0, skipna=True)
            mean_fd = round(column_means['framewise_displacement'], 2)

        return mean_tsnr, mean_fd

    def create_html_viewer(self):
        mi = fsl.MeanImage()
        mean_img_path = os.path.join(self.taskdir, self.task + "_input_functional_bet_mean.nii.gz")
        mi.run(in_file=os.path.join(self.taskdir, self.task + "_input_functional_bet.nii.gz"),
               out_file=mean_img_path)
        html_view = nilearn.plotting.view_img(self.img, threshold=0, bg_img=mean_img_path, vmax=10,
                                              title=self.task)
        html_view.save_as_html(os.path.join(self.outputdir, self.task, self.task + "_viewer.html"))
        viewer_file = "./" + self.task + "/" + self.task + "_viewer.html"
        return viewer_file
