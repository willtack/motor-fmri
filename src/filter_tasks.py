#
# Sometimes, a patient doesn't do one of the tasks. The BIDS layout object gets its task list from the events files present at
# the top level of the directory OR in the func/ directory. So this function gets the task list BEFORE the events files are copied
# into the top level in the run script, so the BIDS layout object is parsing the task list from the files in func/.
#

from bids import BIDSLayout
import argparse
import sys
import flywheel
import json


def main():
    parser = argparse.ArgumentParser('Gets the task list BEFORE the events files are copied into the dataset')
    parser.add_argument(
            "--bidsdir",
            help="Path to a curated BIDS directory",
            required=True
        )

    args = parser.parse_args()
    bidsdir = args.bidsdir

    layout = BIDSLayout(bidsdir)
    task_list = layout.get_tasks()

    # start up inputs
    invocation = json.loads(open('config.json').read())
    config = invocation['config']
    inputs = invocation['inputs']
    destination = invocation['destination']

    fw = flywheel.Client(inputs['api_key']['key'])

    # start up logic:
    analysis_container = fw.get(destination['id'])
    project_container = fw.get(analysis_container.parents['project'])

    for task in task_list:
        fw.download_file_from_project(project_container.id, f"task-{task}_events.tsv", f"/flywheel/v0/input/bids_dataset/task-{task}_events.tsv")

    # remove unhandled tasks
    if 'rest' in task_list:
        task_list.remove('rest')
    if 'imotor2mm' in task_list:
        task_list.remove('imotor2mm')
    if 'imotor3mm' in task_list:
        task_list.remove('imotor3mm')
    if 'motor2mm' in task_list:
        task_list.remove('motor2mm')
    if 'motor3mm' in task_list:
        task_list.remove('motor3mm')

    sys.stdout.write(' '.join(task_list))


if __name__ == "__main__":
    main()
