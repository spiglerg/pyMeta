#!/usr/bin/env python
"""
Build the documentation for pyMeta using Sphinx ("pip install -U sphinx").
Script adapted from Flatland: https://gitlab.aicrowd.com/flatland/flatland
"""
import glob
import os
import shutil
import subprocess
import webbrowser
from urllib.request import pathname2url


def browser(pathname):
    webbrowser.open("file:" + pathname2url(os.path.abspath(pathname)))


def remove_exists(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


# clean docs config and html files, and rebuild everything
# wildcards do not work under Windows
for file in glob.glob(r'./docs/pymeta*.rst'):
    remove_exists(file)
remove_exists('docs/modules.rst')

# copy md files from root folder into docs folder
for file in glob.glob(r'./*.md'):
    print(file)
    shutil.copy(file, 'docs/')

subprocess.call(['sphinx-apidoc', '--force', '-a', '-e', '-o', 'docs/', 'pyMeta', '-H', 'pyMeta Reference'])

os.environ["SPHINXPROJ"] = "pyMeta"
os.chdir('docs')
subprocess.call(['python3', '-msphinx', '-M', 'clean', '.', '_build'])
# TODO fix sphinx warnings instead of suppressing them...
subprocess.call(['python3', '-msphinx', '-M', 'html', '.', '_build'])
# subprocess.call(['python', '-msphinx', '-M', 'html', '.', '_build', '-Q'])

browser('_build/html/index.html')
