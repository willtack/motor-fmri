#
# Sometimes, a patient doesn't do one of the tasks. The BIDS layout object gets its task list from the events files present at
# the top level of the directory OR in the func/ directory. So this function gets the task list BEFORE the events files are copied
# into the top level in the run script, so the BIDS layout object is parsing the task list from the files in func/.
#

from bids import BIDSLayout
import argparse
import sys


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

    # move scenemem to end of list
    if 'scenemem' in task_list:
        task_list.append(task_list.pop(task_list.index('scenemem')))

    # remove unhandled tasks
    if 'rest' in task_list:
        task_list.remove('rest')
    if 'binder' in task_list:
        task_list.remove('binder')
    if 'verbgen' in task_list:
        task_list.remove('verbgen')

    sys.stdout.write(' '.join(task_list))


if __name__ == "__main__":
    main()
