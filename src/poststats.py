#
# Class for computing statistics related to language lateralization and for generating graphs and figures representing these
# statistics, which will be bundled into the final HTML report
#
#
#

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from nipype.algorithms.confounds import TSNR
import os
import nipype.interfaces.fsl as fsl
import nilearn.plotting
import pandas as pd
import seaborn as sns


class PostStats:
    """
    Class to perform analyses on data
    """

    def __init__(self, subject_id, source_img, img, task, run_number, roi_dict_list, confounds, outputdir, datadir):
        self.subject_id = subject_id
        self.source_img = source_img.path
        self.img = img
        self.task = task
        self.run = run_number
        self.outputdir = outputdir
        self.taskdir = os.path.join(outputdir, self.task)
        self.datadir = datadir

        self.roi_dict_list = roi_dict_list  # list of dictionaries with key-value pairs of "ROI label": ['path/to/leftroimask.nii.gz', '/path/to/rightroimask.nii.gz']
        self.temporal_dict = roi_dict_list[0]
        self.frontal_dict = roi_dict_list[1]
        self.misc_dict = roi_dict_list[2]
        self.control_dict = roi_dict_list[3]
        self.motor_dict = roi_dict_list[4]

        self.confounds = confounds  # confounds tsv (returned from setup())
        self.mtl_mask = os.path.join(datadir, "masks", "hpf_bin.nii.gz")

        # Assemble lists
        if self.task == 'motor':
            # Calculate the statistics in each dictionary of regions
            self.left_scene_stats, self.right_scene_stats, self.left_scene_ns, self.right_scene_ns, self.scene_ars = self.calc_stats(self.motor_dict)
            # Define lists for later use constructing tables
            self.left_stats_list = self.left_scene_stats
            self.right_stats_list = self.right_scene_stats
            self.left_ns_list = self.left_scene_ns
            self.right_ns_list = self.right_scene_ns
            self.ars = self.scene_ars
        else:
            # Calculate the statistics in each dictionary of regions
            self.left_temp_stats, self.right_temp_stats, self.left_temp_ns, self.right_temp_ns, self.temp_ars = self.calc_stats( self.temporal_dict)
            self.left_front_stats, self.right_front_stats, self.left_front_ns, self.right_front_ns, self.front_ars = self.calc_stats(self.frontal_dict)
            self.left_misc_stats, self.right_misc_stats, self.left_misc_ns, self.right_misc_ns, self.misc_ars = self.calc_stats(self.misc_dict)
            self.left_ctrl_stats, self.right_ctrl_stats, self.left_ctrl_ns, self.right_ctrl_ns, self.ctrl_ars = self.calc_stats(self.control_dict)
            # Define lists for later use constructing tables
            self.left_stats_list = self.left_temp_stats + self.left_front_stats + self.left_misc_stats + self.left_ctrl_stats
            self.right_stats_list = self.right_temp_stats + self.right_front_stats + self.right_misc_stats + self.right_ctrl_stats
            self.left_ns_list = self.left_temp_ns + self.left_front_ns + self.left_misc_ns + self.left_ctrl_ns
            self.right_ns_list = self.right_temp_ns + self.right_front_ns + self.right_misc_ns + self.right_ctrl_ns
            self.ars = self.temp_ars + self.front_ars + self.misc_ars + self.ctrl_ars

        # # Get list of ROIs. First, loop through the dictionaries
        self.lang_roi_list = []
        for roi_dict in self.roi_dict_list[:-1]:  # all but last item
            self.lang_roi_list = self.lang_roi_list + list(roi_dict.keys())  # then get all the keys, convert to list, and concatenate

        self.scene_roi_list = []
        for roi_dict in self.roi_dict_list[-1]:  # last item
            self.scene_roi_list.append(roi_dict)  # will just be a list with one item

    def create_glass_brain(self):
        nilearn.plotting.plot_glass_brain(nilearn.image.smooth_img(self.img, 4),
                                          output_file=os.path.join(self.outputdir, self.task, 'figs', self.task + "_" + self.run + "_gb.svg"),
                                          display_mode='lyrz', colorbar=True, plot_abs=False, threshold=0)

        out_svg = '"' + '/'.join(('.', self.task, "figs", self.task+ "_" + self.run + "_gb.svg")) + '"'
        return out_svg

    def create_mosaic(self):
        lang_roi = os.path.join(self.datadir, "masks", "lang_roi.nii.gz")
        motor_roi = os.path.join(self.datadir, "masks", "motor_roi.nii.gz")

        if self.task == 'motor':
            display = nilearn.plotting.plot_roi(motor_roi, display_mode='z', cut_coords=8, alpha=0.75, cmap='gray')
            display.add_overlay(self.img, **{"cmap": "cold_hot"})
        else:
            display = nilearn.plotting.plot_roi(lang_roi, display_mode='z', cut_coords=8, alpha=0.75, cmap='gray')
            display.add_overlay(self.img, **{"cmap": "cold_hot"})

        display.savefig(os.path.join(self.outputdir, self.task, 'figs', self.task + "_" + self.run + "_mosaic.png"))
        out_mosaic_svg = '"' + '/'.join(('.', self.task, "figs", self.task + "_" + self.run + "_mosaic.png")) + '"'
        return out_mosaic_svg

    def create_surface(self):
        nilearn.plotting.plot_img_on_surf(self.img,
                                  views=['lateral', 'medial'],
                                  hemispheres=['left', 'right'],
                                  colorbar=True, threshold=1,
                                  output_file=os.path.join(self.outputdir, self.task, 'figs', self.task + "_" + self.run + "_surf.png"))

        out_surf_svg = '"' + '/'.join(('.', self.task, "figs", self.task + "_" + self.run + "_surf.png")) + '"'
        return out_surf_svg

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

    def calc_stats(self, roi_dict):
        """
        Calculate activation statistics in the current list of ROIs
        :param roi_dict: dictionary of ROI labels i.e. str as keys and list of L/R mask files i.e. list[str,str] as values
        :return: left_stats: list of statistics in the left hemisphere ROIs
                 right_stats: list of statistics in the right hemisphere ROIs
                 ars: list of asymmetry ratios for each ROI
        """
        vox = {}  # dictionary of mask: mask voxels
        res = {}  # dictionary of roi: percent activation
        n = {}  # dictionary of n activated voxels in ROI
        left_stats = []  # for plot
        right_stats = []
        leftns = []
        rightns = []
        ars = []  # list of asymmetry ratios for table
        for roi_label in roi_dict:  # loop through the keys (ROI labels)
            mask_list = roi_dict[roi_label]
            for mask in mask_list:
                roi = os.path.basename(mask).split('.')[0]
                vox[roi] = self.get_mask_vox(mask)
                res[roi] = round(self.get_roi_perc(mask, vox[roi]))
                number = res[roi]
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

    def create_bar_plot(self):
        # index = np.arange(len(self.left_stats))
        # bar_width = 0.2
        # opacity = 0.8
        # axes = plt.gca()
        # axes.set_ylim([0, 100])  # it's a percentage
        #
        # plt.bar(index, self.left_stats, bar_width,
        #         alpha=opacity,
        #         color='#4f6bb0',
        #         label='Left')
        # plt.bar(index + bar_width, self.right_stats, bar_width,
        #         alpha=opacity,
        #         color='#550824',
        #         label='Right')
        #
        # plt.xlabel('ROI')
        # plt.ylabel('% activated voxels in ROI')
        # plt.title(self.task)
        # plt.xticks(index + bar_width / 2, self.rois)
        # plt.legend()
        # plt.savefig(os.path.join(self.outputdir, self.task, 'figs', self.task + "_bar.svg"))
        # plt.close()

        # Get table, a pandas Dataframe
        df = self.generate_statistics_table()[1]
        # Plot figure
        if self.task == 'motor':
            plt.figure(figsize=(8, 7))
            a = sns.barplot(x="roi", y="percent", hue="hemisphere", palette=["gray", "firebrick"], data=df)
            a.set_ylabel("Percent Activated Voxels in ROI")
        else:
            f, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=False, sharey=True)
            tmp = sns.barplot(x="roi", y="percent", hue="hemisphere", ax=axes[0, 0], palette=["slategrey", "darkslateblue"], data=df.loc[df['group'] == 'temporal'])
            frnt = sns.barplot(x="roi", y="percent", hue="hemisphere", ax=axes[0, 1], palette=["slategrey", "darkslateblue"], data=df.loc[df['group'] == 'frontal'])
            msc = sns.barplot(x="roi", y="percent", hue="hemisphere", ax=axes[1, 0], palette=["slategrey", "darkslateblue"], data=df.loc[df['group'] == 'misc'])
            ctrl = sns.barplot(x="roi", y="percent", hue="hemisphere", ax=axes[1, 1], palette=["slategrey", "darkslateblue"], data=df.loc[df['group'] == 'control'])
            for a in [frnt, msc, ctrl]:
                a.get_legend().remove()
            for a in [tmp, msc]:
                a.set_ylabel("Percent Activated Voxels in ROI")
            for a in [frnt, ctrl]:
                a.set_ylabel('')

        plt.ylim(0, 100)
        plt.tight_layout()
        # plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.outputdir, self.task, 'figs', self.task + "_" + self.run + "_bar.svg"))
        plt.close()
        plot_file = '"' + '/'.join(('.', self.task, 'figs', self.task + "_" + self.run + "_bar.svg")) + '"'
        return plot_file

    def generate_statistics_table(self):
        # Construct two tables: one to display in html report, one as basis for the bar plot
        if self.task == 'motor':
            left_stats_list = self.left_scene_stats
            right_stats_list = self.right_scene_stats
            ars_list = self.scene_ars
            hem_list = ['left'] * 4 + ['right'] * 4
            group_list = ['motor'] * 8
            list_of_rois = self.scene_roi_list
        else:
            left_stats_list = self.left_temp_stats + self.left_front_stats + self.left_misc_stats + self.left_ctrl_stats
            right_stats_list = self.right_temp_stats + self.right_front_stats + self.right_misc_stats + self.right_ctrl_stats
            ars_list = self.temp_ars + self.front_ars + self.misc_ars + self.ctrl_ars
            hem_list = ['left'] * 14 + ['right'] * 14
            group_list = ['temporal'] * 3 + ['frontal'] * 5 + ['misc'] * 3 + ['control'] * 3
            group_list = group_list * 2
            list_of_rois = self.lang_roi_list

        stats_list = self.left_stats_list + self.right_stats_list
        d4plot = {'roi': list_of_rois * 2, 'percent': stats_list, 'hemisphere': hem_list, 'group': group_list}
        d4table = {'Left %': left_stats_list, 'Right %': right_stats_list, 'Laterality Index': ars_list}
        df4plot = pd.DataFrame(data=d4plot)
        df4table = pd.DataFrame(data=d4table, index=list_of_rois)
        df4table.to_csv(os.path.join(self.outputdir, self.task, 'figs', self.task + "_" + self.run + "_html_table.csv"))
        html_table = df4table.to_html()

        return html_table, df4plot

    def generate_csv(self, left_stats, right_stats, leftns, rightns, ars, rois):
        # table for csv output
        t_snr = self.calc_iqms()[0]
        fd = self.calc_iqms()[1]

        # Create lists of labels for each region: e.g. hemisphere left %, broca's left %, hemisphere right%, broca's right % etc.
        # column_labels = []
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
        row = [self.subject_id] + [self.task] + left_stats + right_stats + ars + leftns + rightns + sums + [fd] + [t_snr]
        data = np.array([row])
        df = pd.DataFrame(data, columns=column_labels)
        df.set_index('subject', inplace=True)
        df.to_csv(os.path.join(self.outputdir, self.task, 'stats', self.task + "_" + self.run + "_data.csv"))

    def generate_csv_wrap(self, task):
        if task == 'motor':
            self.generate_csv(self.left_stats_list, self.right_stats_list, self.left_ns_list, self.right_ns_list, self.ars, self.scene_roi_list)
        else:
            self.generate_csv(self.left_stats_list, self.right_stats_list, self.left_ns_list, self.right_ns_list, self.ars, self.lang_roi_list)

    def calc_iqms(self):
        # tSNR
        tsnr = TSNR()
        tsnr.inputs.in_file = self.source_img
        tsnr.inputs.mean_file = os.path.join(self.outputdir, self.task, self.task + "_" + self.run + "_mean_tsnr.nii.gz")
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
        #mi = fsl.MeanImage()
        #mi_run = mi.run(in_file=os.path.join(self.taskdir, self.task + "_input_functional_masked.nii.gz"))
        #mean_img_path = mi_run.outputs.out_file
        html_view = nilearn.plotting.view_img(self.img, threshold=0, bg_img='MNI152', vmax=10,
                                              title=self.task)
        html_view.save_as_html(os.path.join(self.outputdir, self.task, 'figs', self.task + "_" + self.run + "_viewer.html"))
        viewer_file = '"' + '/'.join(('.', self.task, 'figs', self.task + "_" + self.run +"_viewer.html")) + '"'
        return viewer_file
