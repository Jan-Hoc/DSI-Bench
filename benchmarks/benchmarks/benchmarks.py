# module imports
from dsi_bench.parse import parse_configs
from dsi_bench.datasets import test, oadat, icon, imagenet, shapenet
from dsi_bench.consumers import (
	PythonConsumer,
	TensorflowConsumer,
	PytorchConsumer,
	TensorflowSequenceConsumer,
	DALIPytorchConsumer,
	DALITensorflowConsumer,
)
from dsi_bench.illustrator import Illustrator

# import underlying benchmarking package (https://github.com/Jan-Hoc/PySysLoadBench)
from sysloadbench import Benchmark

# other imports
import os
import json
import torch
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv

# ignore certain tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# check if GPU is available
devices = ['cpu']
if torch.cuda.is_available():
	devices.append('gpu')

# set path for conversions
load_dotenv()
output_path = Path(os.environ.get('TMP_DATA_DIRECTORY', './tmp'))
output_path.mkdir(parents=True, exist_ok=True)

# set defaults for CLI arguments
default_suites = ['TEST', 'ICON', 'OADAT', 'ImageNet', 'ShapeNet']
default_converters = [
	'raw',
	'hdf5',
	'zarr',
	'pickle',
	'npy',
	'npz',
	'tfrecord_single',
	'tfrecord_multi',
]
default_consumers = [
	'python',
	'tensorflow',
	'tensorflow_sequence',
	'pytorch',
	'dali_pytorch',
	'dali_tensorflow',
]
default_return_types = ['python', 'tensorflow', 'pytorch']

# set up different CLI options
parser = argparse.ArgumentParser(prog='loader-benchmark', description='Manual benchmark')
parser.add_argument(
	'--suites', default=default_suites, nargs='*'
)  # configuration files to include in benchmark
parser.add_argument(
	'--converter_names',
	default=default_converters,
	choices=default_converters,
	nargs='*',
)  # storage formats to include in benchmark
parser.add_argument(
	'--consumers', default=default_consumers, choices=default_consumers, nargs='*'
)  # data formats to include in benchmark
parser.add_argument(
	'--devices', default=devices, choices=devices, nargs='*'
)  # run on cpu or gpu (if available)
parser.add_argument(
	'--return_types',
	default=default_return_types,
	choices=default_return_types,
	nargs='*',
)  # return types to include in benchmark
parser.add_argument('--name_prefix', default='')  # prefix to add to suite_name
parser.add_argument(
	'--rounds', type=int, default=3, choices=range(1, 1000)
)  # amount of rounds to execute benchmark rounds for
parser.add_argument(
	'--delete_conversions', action='store_true'
)  # keep or delete conversions after benchmark suite
parser.add_argument('--keep_conversions', dest='delete_conversions', action='store_false')
parser.set_defaults(delete_conversions=True)

args, unknown = parser.parse_known_args()

# parse the different suite configs
configs = parse_configs()


# remove keys from configs that are not passed to data set classes
def clean_config(config: dict) -> dict:
	data_source_args = {}
	remove_keys = ['suite_name', 'dataset_type', 'batch_size', 'num_threads']

	for key, value in config.items():
		if key not in remove_keys:
			data_source_args[key] = value

	return data_source_args


# calculate size of directory containing data set conversion
def directory_size(location: str) -> float:
	disk_space_used = os.popen(f'du -bs {location} | grep -o "^[0-9]*"').read()

	if disk_space_used == '':
		disk_space_used = '4096'

	# subtract size of directory and convert to MiB
	disk_space_used = (int(disk_space_used) - 4096) / (1024**2)
	return disk_space_used


# run actual benchmark
if __name__ == '__main__':
	consumer_lut = {
		'python': PythonConsumer,
		'tensorflow': TensorflowConsumer,
		'tensorflow_sequence': TensorflowSequenceConsumer,
		'pytorch': PytorchConsumer,
		'dali_pytorch': DALIPytorchConsumer,
		'dali_tensorflow': DALITensorflowConsumer,
	}

	data_sources_lut = {
		'test': test,
		'oadat': oadat,
		'icon': icon,
		'imagenet': imagenet,
		'shapenet': shapenet,
	}

	# iterate through the different suites
	for suite_name in args.suites:
		if suite_name not in configs:
			raise KeyError(
				'config for data source with given name ' + suite_name + ' not found'
			)

		print(f'\n\n\nStarting benchmarks for suite {suite_name}\n')

		used_diskspace = {}

		# create benchmark instance for this suite
		suite_name_prefixed = f'{args.name_prefix}{suite_name}'
		benchmark = Benchmark(suite_name_prefixed)

		# get configuration of current suite
		config = configs[suite_name]
		data_source_args = clean_config(config)

		# create according DataSource for used data set
		current_data_source = data_sources_lut[config['dataset_type']]
		base_data_source = current_data_source.DataSource(**data_source_args)
		base_data_source.prepare()

		#####################################
		##### RUNS FOR NON-DALI LOADERS #####
		#####################################

		# iterate through selected file formats
		converters = current_data_source.get_available_converters()
		for converter_name in args.converter_names:
			# selected conversion not available
			if converter_name not in converters:
				continue

			print(
				f'\nBenchmarking suite {suite_name} for data set {config["dataset_type"]} converted to {converter_name}\n'
			)

			# where to save conversion
			conversion_path = output_path / suite_name / converter_name

			# iterate through selected non-dali data loaders
			for consumer_name in args.consumers:
				# technical issues with these combinations in gpu environments, so skip
				if ('ICON' in suite_name or 'ShapeNet' in suite_name) and 'tensorflow' in consumer_name and 'tfrecord' in converter_name and 'gpu' in devices:
					continue

				# dali loaders handled separately later
				if 'dali' in consumer_name:
					continue

				print(f'Using dataloader {consumer_name.upper()}')

				# iterate through selected return types
				for return_type in args.return_types:
					# incompatible combination, skip
					if (return_type == 'tensorflow' and consumer_name == 'pytorch') or (
						return_type not in base_data_source.get_available_return_types()
					):
						continue

					# run benchmark run for selected devices
					for device in args.devices:
						# incompatible combination, skip
						# don't necessarily return tensor so cant load to gpu
						if device == 'gpu' and consumer_name == 'python':
							continue

						# function to run before rounds of run
						# makes sure conversion was done
						def setup_func(
							base_data_source,
							converter,
							conversion_path,
							consumer_lut,
							consumer_name,
							config,
							return_type,
							device,
						):
							converter(base_data_source, conversion_path)

						# function to run before every round of run
						# sets up data loader to consume converted data set
						def prerun_func(
							base_data_source,
							converter,
							conversion_path,
							consumer_lut,
							consumer_name,
							config,
							return_type,
							device,
						):
							global consumer
							converted_data_source = converter(base_data_source, conversion_path)
							converted_data_source.set_return_type(return_type)

							consumer = consumer_lut[consumer_name](
								dataset=converted_data_source,
								num_threads=config['num_threads'],
								batch_size=config['batch_size'],
								device=device,
							)

						# function being benchmarked, actual data set consumption
						def benchmark_func(
							base_data_source,
							converter,
							conversion_path,
							consumer_lut,
							consumer_name,
							config,
							return_type,
							device,
						):
							global consumer
							consumer()

						# surround in try except so in case of error other runs can still execute
						try:
							# run benchmark
							benchmark.add_run(
								f'{converter_name} {consumer_name} {device} {return_type}',
								benchmark_func,
								setup=setup_func,
								prerun=prerun_func,
								rounds=args.rounds,
								kwargs={
									'base_data_source': base_data_source,
									'converter': converters[converter_name],
									'conversion_path': conversion_path,
									'consumer_lut': consumer_lut,
									'consumer_name': consumer_name,
									'config': config,
									'return_type': return_type,
									'device': device,
								},
							)
						except Exception as e:
							print(e)

			# read disk space required by conversion
			used_diskspace[converter_name] = round(
				directory_size(conversion_path.absolute().as_posix()), 2
			)

			print(
				f'\nSpace on disk needed for converted data set {config["dataset_type"]} of suite {suite_name} in format {converter_name}: {used_diskspace[converter_name]}MiB\n'
			)

		#####################################
		####### RUNS FOR DALI LOADERS #######
		#####################################

		# get defined DALI pipelines of data set
		pipelines = (
			current_data_source.get_available_pipelines(base_data_source.key)
			if hasattr(base_data_source, 'key')
			else current_data_source.get_available_pipelines()
		)

		# iterate through available pipelines
		for data_format, dali_params in pipelines.items():
			conversion_path = output_path / suite_name / data_format

			# pipeline not of interest since doesn't match any specified storage format
			if data_format not in args.converter_names:
				continue

			for i, consumer_name in enumerate(args.consumers):
				# ignore non-dali loader
				if 'dali' not in consumer_name:
					continue

				if i == 0:
					print(
						f'\nBenchmarking suite {suite_name} for data set {config["dataset_type"]} using DALI converted to {data_format}\n'
					)

				print(f'Using dataloader {consumer_name.upper()}')

				# run benchmark run for selected devices
				for device in args.devices:
					# function to run before rounds of run
					# makes sure conversion was done and pipeline arguments are available
					def setup_func(
						base_data_source,
						consumer_lut,
						consumer_name,
						conversion_path,
						dali_params,
						config,
						device,
					):
						global pipe_args

						base_data_source.set_return_type(
							'python'
						)  # irrelevant for dali so set to "default"
						pipe_args = dali_params['preparation'](base_data_source, conversion_path)

					# function to run before every round of run
					# sets up data loader to consume converted data set and passes according arguments
					def prerun_func(
						base_data_source,
						consumer_lut,
						consumer_name,
						conversion_path,
						dali_params,
						config,
						device,
					):
						global consumer, pipe_args

						# pass specific arguments to pytorch dali loader
						if consumer_name == 'dali_pytorch':
							labels = []
							if dali_params['labels'] is not None:
								labels = dali_params['labels']
							else:
								labels = base_data_source.key

							consumer = consumer_lut[consumer_name](
								pipeline=dali_params['pipeline'],
								pipe_args=pipe_args,
								labels=labels,
								num_threads=config['num_threads'],
								batch_size=config['batch_size'],
								device=device,
								**dali_params['pytorch_args'],
							)
						# pass specific arguments to tensorflow dali loader
						elif consumer_name == 'dali_tensorflow':
							consumer = consumer_lut[consumer_name](
								pipeline=dali_params['pipeline'],
								pipe_args=pipe_args,
								num_threads=config['num_threads'],
								batch_size=config['batch_size'],
								device=device,
								**dali_params['tensorflow_args'],
							)
						# default case if other dali loaders get added
						else:
							consumer = consumer_lut[consumer_name](
								pipeline=dali_params['pipeline'],
								pipe_args=pipe_args,
								num_threads=config['num_threads'],
								batch_size=config['batch_size'],
								device=device,
							)

					# function being benchmarked, actual data set consumption
					def benchmark_func(
						base_data_source,
						consumer_lut,
						consumer_name,
						conversion_path,
						dali_params,
						config,
						device,
					):
						global consumer
						consumer()

					# surround in try except so in case of error other runs can still execute
					try:
						benchmark.add_run(
							f'{data_format} {consumer_name} {device}',
							benchmark_func,
							setup=setup_func,
							prerun=prerun_func,
							rounds=args.rounds,
							kwargs={
								'base_data_source': base_data_source,
								'consumer_lut': consumer_lut,
								'consumer_name': consumer_name,
								'conversion_path': conversion_path,
								'dali_params': dali_params,
								'config': config,
								'device': device,
							},
						)
					except Exception as e:
						print(e)

		# if there where any successful runs, generate result graphs and save to results/<suite_name>
		if len(benchmark.statistics()) > 0:
			benchmark.save_results()
			path = Path.cwd() / 'results' / suite_name_prefixed
			path.mkdir(parents=True, exist_ok=True)

			result_data = {
				'system_information': benchmark.get_sysinfo(),
				'run_results': benchmark.statistics(),
				'used_diskspace': used_diskspace,
			}

			with open(path / 'raw_results.json', 'w') as result_file:
				json.dump(result_data, result_file, indent=4)

			Illustrator.illustrate_results(path, result_data, suite_name)

		# do clean up if necessary
		if args.delete_conversions:
			try:
				shutil.rmtree(output_path / suite_name, ignore_errors=True)
			except Exception:
				pass
