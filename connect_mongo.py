#!/usr/bin/env python

##########################################################################
# Copyright 2018 Kata.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import argparse
import os

from pymongo import MongoClient


def main(args):
    client = MongoClient(args.url)
    db = client[args.db_name]
    try:
        from IPython import start_ipython
        start_ipython(argv=[], user_ns=dict(db=db))
    except ImportError:
        import code
        shell = code.InteractiveConsole(dict(db=db))
        shell.interact()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Connect to Sacred's MongoDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--url', default=os.getenv('SACRED_MONGO_URL'), help='the mongo url')
    parser.add_argument('--db-name', default=os.getenv('SACRED_DB_NAME'), help='the db name')
    args = parser.parse_args()
    main(args)
