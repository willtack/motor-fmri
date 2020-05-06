import os
import nipype.algorithms.modelgen as model
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
import nibabel
from nistats import thresholding

datadir = os.getcwd()
template = os.path.join(datadir, "masks", "mni152.nii.gz")


def model_fitting(source_img, prepped_img, subject_info, aroma, task, args, mask_file, run_number):
    # Get the necessary parameters
    outputdir = args.outputdir
    fwhm = args.fwhm
    cthresh = args.cthresh
    alpha = args.alpha

    # Make a task directory in the output folder
    if run_number > 0:
        taskdir = os.path.join(outputdir, task + "_run-0" + str(run_number + 1))
    else:
        taskdir = os.path.join(outputdir, task)

    if not os.path.exists(taskdir):
        os.mkdir(taskdir)
    os.mkdir(os.path.join(taskdir, 'stats'))
    os.mkdir(os.path.join(taskdir, 'figs'))

    processed_image = preprocess(aroma, fwhm, prepped_img, mask_file, taskdir, task)

    task_vs_baseline = [task + " vs baseline", 'T', [task, 'baseline'], [1, -1]]  # set up contrasts
    contrasts = [task_vs_baseline]

    """
    Model fitting workflow

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
         outputspec.zfiles
         outputspec.parameter_estimates
    """

    modelfit = pe.Workflow(name='modelfit', base_dir=taskdir)
    modelspec = pe.Node(interface=model.SpecifyModel(), name="modelspec")  # generate design info
    inputspec = pe.Node(util.IdentityInterface(
        fields=['session_info', 'interscan_interval', 'contrasts', 'film_threshold', 'functional_data', 'bases',
                'model_serial_correlations']), name='inputspec')
    level1design = pe.Node(interface=fsl.Level1Design(), name="level1design")
    modelgen = pe.MapNode(interface=fsl.FEATModel(), name='modelgen', iterfield=['fsf_file', 'ev_files'])
    modelestimate = pe.MapNode(interface=fsl.FILMGLS(smooth_autocorr=True, mask_size=5), name='modelestimate',
                               iterfield=['design_file', 'in_file', 'threshold'])
    glm = pe.MapNode(interface=fsl.GLM(mask=mask_file, output_type='NIFTI_GZ', out_z_name=os.path.join(taskdir, task + '_z.nii.gz')), name='glm', iterfield=['contrasts', 'in_file', 'design'])
    outputspec = pe.Node(
        util.IdentityInterface(fields=['copes', 'varcopes', 'dof_file', 'zfiles', 'parameter_estimates']),
        name='outputspec')

    modelfit.connect([
        (modelspec, inputspec, [('session_info', 'session_info')]),
        (inputspec, level1design,
         [('interscan_interval', 'interscan_interval'), ('session_info', 'session_info'), ('contrasts', 'contrasts'),
          ('bases', 'bases'), ('model_serial_correlations', 'model_serial_correlations')]),
        (inputspec, modelestimate, [('film_threshold', 'threshold'), ('functional_data', 'in_file')]),
        (inputspec, glm, [('functional_data', 'in_file')]),
        (level1design, modelgen, [('fsf_files', 'fsf_file'), ('ev_files', 'ev_files')]),
        (modelgen, modelestimate, [('design_file', 'design_file')]),
        (modelgen, glm, [('con_file', 'contrasts'), ('design_file', 'design')]),
        (glm, outputspec, [('out_cope', 'copes'), ('out_varcb', 'varcopes'), ('out_z', 'zfiles')]),
        (modelestimate, outputspec, [('param_estimates', 'parameter_estimates'), ('dof_file', 'dof_file')])
    ])

    # Define inputs to workflow
    modelspec.inputs.functional_runs = processed_image
    inputspec.inputs.functional_data = processed_image
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
    output_txt = open(os.path.join(taskdir, task + '_outputs.txt'), 'w')
    print_outputs(output_txt, res)

    z_img = os.path.join(taskdir, task + '_z.nii.gz')

    # Use False Discovery Rate theory to correct for multiple comparisons
    fdr_thresh_img, fdr_threshold = thresholding.map_threshold(stat_img=z_img,
                                                               mask_img=mask_file,
                                                               alpha=alpha,
                                                               height_control='fdr',
                                                               cluster_threshold=cthresh)
    print("Thresholding at FDR corrected threshold of " + str(fdr_threshold))
    fdr_thresh_img_path = os.path.join(taskdir, task + '_fdr_thresholded_z.nii.gz')
    nibabel.save(fdr_thresh_img, fdr_thresh_img_path)

    # Do a cluster analysis using the FDR corrected threshold on the original z_img
    print("Performing cluster analysis.")
    cl = fsl.Cluster(in_file=z_img, threshold=fdr_threshold)
    cluster_file = os.path.join(taskdir, 'stats', task + "_cluster_stats.txt")
    cluster_analysis(cluster_file, cl)

    # Resample the result image with AFNI
    resample_fdr_thresh_img_path = os.path.join(taskdir, task + '_fdr_thresholded_z_resample.nii.gz')
    print("Resampling thresholded image to MNI space")
    resample = afni.Resample(master=template, out_file=resample_fdr_thresh_img_path, in_file=fdr_thresh_img_path)
    resample.run()
    os.remove(fdr_thresh_img_path)

    print("Image to be returned: " + resample_fdr_thresh_img_path)

    return resample_fdr_thresh_img_path


def print_outputs(file, workflow_results):
    print('', file=file)
    print(list(workflow_results.nodes), file=file)
    print('', file=file)
    for i in range(len(list(workflow_results.nodes)) - 1):
        print("%s: %s" % ("NODE", list(workflow_results.nodes)[i]), file=file)
        print(list(workflow_results.nodes)[i].result.outputs, file=file)
        print('', file=file)
    file.close()


def cluster_analysis(cluster_file, cluster_run):
    cluster_results = cluster_run.run()
    clusters = cluster_results.runtime.stdout  # write the terminal output to a text file
    f = open(cluster_file, 'w')
    f.write(clusters)
    f.close()


def preprocess(aroma_config, fwhm_config, input_img, mask_file, taskdir, task):
    # Resample to mask space
    # This is for AROMA images. Does it hurt non-AROMA?
    resample_path = os.path.join(taskdir, task + '_resample.nii.gz')
    resample = afni.Resample(master=mask_file, out_file=resample_path, in_file=input_img)
    resample_run = resample.run()

    # Apply fmriprep-calculated brain mask to the functional image
    masked_file_path = os.path.join(taskdir, task + "_input_functional_masked.nii.gz")
    applymask = fsl.ApplyMask(mask_file=mask_file, out_file=masked_file_path)
    applymask.inputs.in_file = resample_run.outputs.out_file
    mask_run = applymask.run()
    masked_input = masked_file_path

    if aroma_config:
        print("No smoothing required.")
        processed_image = masked_input
    else:
        # smoothing
        print("Smoothing the skull-stripped BOLD.")
        smooth = fsl.Smooth(in_file=masked_input, fwhm=fwhm_config)
        smooth_run = smooth.run()
        processed_image = smooth_run.outputs.smoothed_file

    return processed_image
