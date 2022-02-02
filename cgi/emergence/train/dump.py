from typing import Dict, Any
import json

def dump_params(opts: Dict[str, Any]):
    assert isinstance(opts, dict), opts
    excludes = [
        "mode",
    ]
    dump = {"mode": "config"}
    for k in opts.keys():
        if k not in excludes:
            dump[k] = opts[k]
    print(json.dumps(dump, default=repr))
