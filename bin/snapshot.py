#!/usr/bin/env python
import argparse
import glob
import subprocess
from functools import reduce
from subprocess import check_call
from datetime import datetime
import os
import shutil

def multi_glob(files):
    if isinstance(files, str):
        return glob.glob(files)
    files = [set(glob.glob(x)) for x in files]
    return reduce(lambda x, y: x | y, files, set())


def convert_from_ipynb(file, formats, target_dir=None):
    file_dir = os.path.dirname(file) or '.'
    if target_dir is None:
        target_dir = file_dir
    for fmt in formats:
        # print(['jupyter', 'nbconvert', '--to', fmt, file, '--output-dir', target_dir], file_dir)
        check_call(['jupyter', 'nbconvert', '--to', fmt, file, '--output-dir', target_dir], cwd=file_dir)


def snapshot(files='*ipynb', formats=[], message=""):
    found_files = multi_glob(files)
    if len(found_files) == 0:
        print("No file found. Not doing anything.")
        return
    timestamp_date = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")
    target_dir_name = os.path.join("snapshots/", timestamp_date)
    print(f"Dir name: {target_dir_name}")
    print(f"Formats: {formats}")
    os.makedirs(target_dir_name, exist_ok=True)
    for file in found_files:
        if file.startswith('snapshot_'):
            continue
        print(f"File: {file}")
        target = os.path.join(target_dir_name, file)
        # print(target)
        if os.path.isdir(file):
            shutil.copytree(file, target)
        else:
            shutil.copy(file, target)
        if file.endswith('.ipynb'):
            try:
                convert_from_ipynb(file, formats, target_dir=target_dir_name)
            except subprocess.CalledProcessError as e:
                print(e)
    print(message)
    subprocess.getoutput(f"echo {message} > {target_dir_name}/message.txt")
    subprocess.getoutput(f"git log >> {target_dir_name}/message.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', dest='files', nargs='*',
                        default="./*ipynb", help="Files to take snapshot from.")
    parser.add_argument('--formats', dest='formats', nargs='*',
                        default=['script', 'html'], help="formats to save ipynb notebooks in.")
    parser.add_argument('--message', '-m', dest='message', nargs='*',
                        default=[], help="snapshot message")
    args = parser.parse_args()
    formats = args.formats
    files = args.files
    message = ' '.join(args.message)

    snapshot(files, formats, message)
