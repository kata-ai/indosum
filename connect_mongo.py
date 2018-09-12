#!/usr/bin/env python

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
