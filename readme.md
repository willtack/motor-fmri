# lingua-map
*started August 13, 2019*

lingua-map is a program for automatically generating an html report summarizing the results of a patients' presurgical fMRI. The report contains resulting activation maps and ROI statistics. It was originally designed for epilepsy patients who receive an fMRI to lateralize language function prior to surgery. 

The algorithm is Dockerized for use as a gear on the Flywheel neuroinformatics platform. As of now, it is not recommended for use outside of Flywheel.

The program takes a BIDs dataset and fMRIPREP results as inputs.It does a general linear model (GLM) analysis of the preprocessed BOLD time series images to create activation maps using FSL commands wrapped in nipype, then calculates laterality statistics in language ROIs informed by the ASFNR white paper: http://www.ajnr.org/content/38/10/E65. From these results, it generates an html report using jinja-based templating. 
