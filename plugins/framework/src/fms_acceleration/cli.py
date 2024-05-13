# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import sys
from typing import List

from .constants import PLUGIN_PREFIX, PLUGINS

from pip._internal.cli.main import main as pipmain

GITHUB_URL = "github.com/foundation-model-stack/fms-acceleration.git"

# TODO: make a version that fetches the
def install_plugin(
    *args: List[str],
):
    "function to install plugin. pkg_name_or_path can be a name or local path"

    pkg_name_or_path = [x for x in args if not x.startswith('-')]
    assert len(pkg_name_or_path) == 1,\
        "Please specify exactly one plugin to install"
    pkg_name_or_path = pkg_name_or_path[0]

    # take the flags
    args = [x for x in args if x.startswith('-')]

    if os.path.exists(pkg_name_or_path):
        pipmain(['install', *args, pkg_name_or_path])
        return 

    if not pkg_name_or_path.startswith(PLUGIN_PREFIX):
        pkg_name_or_path = f"{PLUGIN_PREFIX}{pkg_name_or_path}"

    # otherwise should be an internet install
    pipmain([
        'install', *args, 
        f'git+https://{GITHUB_URL}#subdirectory=plugins/accelerated-{pkg_name_or_path}' 
    ])

def list_plugins():
    print(
        "\nChoose from the list of plugin shortnames, and do:\n"
        " * 'python -m fms_acceleration.cli install <pip-install-flags> PLUGIN_NAME'.\n\n"
        "Alternatively if the repository was checked out, pip install it from REPO_PATH:\n"
        " * 'pip install <pip-install-flags> REPO_PATH/plugins/PLUGIN_NAME'.\n\n"
        "List of PLUGIN_NAME [PLUGIN_SHORTNAME]:\n"
    )
    for i, name in enumerate(PLUGINS):
        print(f"{i+1}. {PLUGIN_PREFIX}{name} [{name}]")

def cli():
    # not using argparse since its so simple
    message = (
        "FMS Acceleration Framework Command Line Tool.\n"
        "Command line tool to help manage the Acceleration Framework packages.\n"
    )
    argv = sys.argv
    if len(argv) == 1:
        print (message)
        return
    else:
        command = argv[1]
        if len(argv) > 2:
            variadic = sys.argv[2:]
        else:
            variadic = []

    if command == 'install':
        assert len(variadic) >= 1, "Please provide the acceleration plugin name"
        install_plugin(*variadic)
    elif command == 'list':
        assert len(variadic) == 0, "list does not require arguments"
        list_plugins()
    else:
        raise NotImplementedError(
            f"Unknown fms_acceleration.cli command '{command}'"
        )

if __name__ == '__main__':
    cli()