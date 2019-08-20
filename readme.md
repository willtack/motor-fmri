# presurgical-report
*started August 13, 2019*

presurgical-report is a nipype-based program for automatically generating a report (using jinja) of a patients' presurgical fMRI. The report contains resulting activation maps and ROI statistics. It was originally designed for presurgical epilepsy patients who receive an fMRI to lateralize language function. 

The algorithm is Dockerized with the intent of turning it into a gear for the Flywheel neuroinformatics platform. 

The program takes a BIDs dataset and fMRIPREP results as inputs. It does a general linear model (GLM) analysis of the preprocessed BOLD time series images to create activation maps using FSL commands wrapped in nipype.  From these results, it generates an html report using jinja-based templating. 
