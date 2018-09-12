#!/usr/bin/env python

from concurrent.futures import ThreadPoolExecutor as Executor
import argparse
import json
import sys
import typing

from pythonrouge.pythonrouge import Pythonrouge

from data import Document


def label_sentences(doc: Document,
                    tol: float = 0.01,
                    ) -> typing.Tuple[float, Document]:
    if doc.summary is None:
        raise TypeError('document must have gold summary')

    doc_sents = [' '.join(sent) for sent in doc.sentences]
    ref_sents = [' '.join(sent) for sent in doc.summary]

    def compute_rouge(candidate: typing.List[int]) -> float:
        hypothesis = [doc_sents[sid] for sid in sorted(candidate)]
        rouge = Pythonrouge(
            summary_file_exist=False, summary=[hypothesis], reference=[[ref_sents]],
            stemming=False, ROUGE_SU4=False)
        score = rouge.calc_score()
        return score['ROUGE-1-F']

    current: typing.List[int] = []
    best_rouge = -float('inf')
    pool = set(range(len(doc_sents)))
    while pool:
        rouge_scores = [(compute_rouge(current + [sid]), sid) for sid in pool]
        max_rouge, max_rouge_sid = max(rouge_scores, key=lambda pair: pair[0])
        if max_rouge >= best_rouge + tol:
            best_rouge = max_rouge
            current.append(max_rouge_sid)
            pool.remove(max_rouge_sid)
        else:
            break

    current_set = set(current)
    sid = 0
    for para in doc:
        for sent in para:
            sent.label = sid in current_set
            sid += 1
    return best_rouge, doc


def main(args):
    docs = []
    with open(args.path, encoding=args.encoding) as f:
        for linum, line in enumerate(f):
            try:
                obj = json.loads(line.strip())
                docs.append(Document.from_mapping(obj))
            except Exception as e:
                message = f'line {linum+1}: {e}'
                raise RuntimeError(message)

    with Executor(max_workers=args.max_workers) as ex:
        results = ex.map(label_sentences, docs)
        for best_rouge, doc in results:
            print(json.dumps(doc.to_dict(), sort_keys=True))
            if args.verbose:
                print(f'ROUGE-1-F: {best_rouge:.2f}', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Make oracles from a JSONL file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='path to the JSONL file')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    parser.add_argument(
        '-w', '--max-workers', type=int, default=20, help='max number of worker threads')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        parser.error(str(e))
