#!/usr/bin/env python
#
# Authors: Tinashe Tapera, Matt Cieslak
#

import json
import flywheel
import logging
from fw_heudiconv.cli import export


# logging stuff
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fw-heudiconv-gear')
logger.info("=======: fw-heudiconv starting up :=======")

# start up inputs
invocation = json.loads(open('config.json').read())
config = invocation['config']
inputs = invocation['inputs']
destination = invocation['destination']

fw = flywheel.Client(inputs['api_key']['key'])
user = fw.get_current_user()

# start up logic:
heuristic = None  # inputs['heuristic']['location']['path']
analysis_container = fw.get(destination['id'])
project_container = fw.get(analysis_container.parents['project'])
project_label = project_container.label
dry_run = False  # config['dry_run']
action = "Export"  # config['action']

# whole project, single session?
do_whole_project = False  # config['do_whole_project']

if not do_whole_project:

    # find session object origin
    session_container = fw.get(analysis_container.parent['id'])
    sessions = [session_container.label]
    # find subject object origin
    subject_container = fw.get(session_container.parents['subject'])
    subjects = [subject_container.label]

else:
    sessions = None
    subjects = None

# logging stuff
logger.info("Running fw-heudiconv with the following settings:")
logger.info("Project: {}".format(project_label))
logger.info("Subject(s): {}".format(subjects))
logger.info("Session(s): {}".format(sessions))
logger.info("Action: {}".format(action))
logger.info("Dry run: {}".format(dry_run))

# action
if action == "Curate":
    print("This gear does not do BIDS Curation!")

elif action == "Export":

    downloads = export.gather_bids(fw, project_label, subjects, sessions)
    export.download_bids(fw, downloads, "/flywheel/v0/input", dry_run=dry_run)
    if not dry_run:
        pass

elif action == "Tabulate":
    print("This gear does not do BIDS Tabulation!")
else:
    raise Exception('Action not specified correctly!')
