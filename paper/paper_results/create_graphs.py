from dsi_bench.illustrator import Illustrator
from pathlib import Path
import argparse
import json
import os

# set available directories as default
default_suites = [f.name for f in os.scandir(os.path.dirname(__file__)) if f.is_dir()]

parser = argparse.ArgumentParser(
	prog='create_grapth', description='Create graphs from benchmark results'
)
parser.add_argument(
	'--suites', default=default_suites, choices=default_suites, nargs='*'
)

args, unknown = parser.parse_known_args()

# create graphs for selected suites
for res_dir in args.suites:
	res_dir = Path(os.path.dirname(__file__)) / res_dir
	res_file = res_dir / 'raw_results.json'

	if not res_file.is_file():
		continue

	with open(res_file) as f:
		raw_results = json.load(f)
		Illustrator.illustrate_results(
			res_dir, raw_results, res_dir.name[res_dir.name.rfind('_') + 1 :]
		)
