import json
import gzip


def load_jsonl_gz(filepath):
    ret = []
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            ret.append(json.loads(line))
    return ret