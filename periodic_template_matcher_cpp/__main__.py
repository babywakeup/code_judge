import argparse
import json
from .api import run_config


def main():
    parser=argparse.ArgumentParser(description='C++ periodic matcher with Python IO')
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=run_config(args.config,args.out)
    print(json.dumps({k:result[k] for k in ('accepted','status','best_template_index')},ensure_ascii=False))


if __name__=='__main__':main()
