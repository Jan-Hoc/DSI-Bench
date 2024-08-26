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
default_executions = ['TEST', 'ICON', 'OADAT', 'ImageNet', 'ShapeNet']
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
default_data_loaders = [
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
	'--executions', default=default_executions, nargs='*'
)  # configuration files to include in benchmark
parser.add_argument(
	'--converter_names',
	default=default_converters,
	choices=default_converters,
	nargs='*',
)  # storage formats to include in benchmark
parser.add_argument(
	'--data_loaders', default=default_data_loaders, choices=default_data_loaders, nargs='*'
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
parser.add_argument('--name_prefix', default='')  # prefix to add to execution_name
parser.add_argument(
	'--rounds', type=int, default=3, choices=range(1, 1000)
)  # amount of rounds to execute benchmark rounds for
parser.add_argument(
	'--delete_conversions', action='store_true'
)  # keep or delete conversions after benchmark execution
parser.add_argument('--keep_conversions', dest='delete_conversions', action='store_false')
parser.set_defaults(delete_conversions=True)

args, unknown = parser.parse_known_args()

# parse the different execution configs
configs = parse_configs()


# remove keys from configs that are not passed to data set classes
def clean_config(config: dict) -> dict:
	data_source_args = {}
	remove_keys = ['execution_name', 'dataset_type', 'batch_size', 'num_threads']

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
	data_loader_lut = {
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

	# iterate through the different executions
	for execution_name in args.executions:
		if execution_name not in configs:
			raise KeyError(
				'config for data source with given name ' + execution_name + ' not found'
			)

		print(f'\n\n\nStarting benchmarks for configuration {execution_name}\n')

		used_diskspace = {}

		# create benchmark instance for this execution
		execution_name_prefixed = f'{args.name_prefix}{execution_name}'
		benchmark = Benchmark(execution_name_prefixed)

		# get configuration of current execution
		config = configs[execution_name]
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
				f'\nBenchmarking execution {execution_name} for data set {config["dataset_type"]} converted to {converter_name}\n'
			)

			# where to save conversion
			conversion_path = output_path / execution_name / converter_name

			# iterate through selected non-dali data loaders
			for data_loader_name in args.data_loaders:
				# dali loaders handled seperately later
				if 'dali' in data_loader_name:
					continue

				print(f'Using dataloader {data_loader_name.upper()}')

				# iterate through selected return types
				for return_type in args.return_types:
					# incompatible combination, skip
					if (return_type == 'tensorflow' and data_loader_name == 'pytorch') or (
						return_type not in base_data_source.get_available_return_types()
					):
						continue

					# run benchmark run for selected devices
					for device in args.devices:
						# incompatible combination, skip
						# don't necessarily return tensor so cant load to gpu
						if device == 'gpu' and data_loader_name == 'python':
							continue

						# function to run before rounds of run
						# makes sure conversion was done
						def setup_func(
							base_data_source,
							converter,
							conversion_path,
							data_loader_lut,
							data_loader_name,
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
							data_loader_lut,
							data_loader_name,
							config,
							return_type,
							device,
						):
							global data_loader
							converted_data_source = converter(base_data_source, conversion_path)
							converted_data_source.set_return_type(return_type)

							data_loader = data_loader_lut[data_loader_name](
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
							data_loader_lut,
							data_loader_name,
							config,
							return_type,
							device,
						):
							global data_loader
							data_loader()

						# surround in try except so in case of error other runs can still execute
						try:
							# run benchmark
							benchmark.add_run(
								f'{converter_name} {data_loader_name} {device} {return_type}',
								benchmark_func,
								setup=setup_func,
								prerun=prerun_func,
								rounds=args.rounds,
								kwargs={
									'base_data_source': base_data_source,
									'converter': converters[converter_name],
									'conversion_path': conversion_path,
									'data_loader_lut': data_loader_lut,
									'data_loader_name': data_loader_name,
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
				f'\nSpace on disk needed for converted data set {config["dataset_type"]} of execution {execution_name} in format {converter_name}: {used_diskspace[converter_name]}MiB\n'
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
			conversion_path = output_path / execution_name / data_format

			# pipeline not of interest since doesn't match any specified storage format
			if data_format not in args.converter_names:
				continue

			for i, data_loader_name in enumerate(args.data_loaders):
				# ignore non-dali loader
				if 'dali' not in data_loader_name:
					continue

				if i == 0:
					print(
						f'\nBenchmarking execution {execution_name} for data set {config["dataset_type"]} using DALI converted to {data_format}\n'
					)

				print(f'Using dataloader {data_loader_name.upper()}')

				# run benchmark run for selected devices
				for device in args.devices:
					# function to run before rounds of run
					# makes sure conversion was done and pipeline arguments are available
					def setup_func(
						base_data_source,
						data_loader_lut,
						data_loader_name,
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
						data_loader_lut,
						data_loader_name,
						conversion_path,
						dali_params,
						config,
						device,
					):
						global data_loader, pipe_args

						# pass specific arguments to pytorch dali loader
						if 'pytorch' in data_loader_name:
							labels = []
							if dali_params['labels'] is not None:
								labels = dali_params['labels']
							else:
								labels = base_data_source.key

							data_loader = data_loader_lut[data_loader_name](
								pipeline=dali_params['pipeline'],
								pipe_args=pipe_args,
								labels=labels,
								num_threads=config['num_threads'],
								batch_size=config['batch_size'],
								device=device,
								**dali_params['pytorch_args'],
							)
						# pass specific arguments to tensorflow dali loader
						elif 'tensorflow' in data_loader_name:
							data_loader = data_loader_lut[data_loader_name](
								pipeline=dali_params['pipeline'],
								pipe_args=pipe_args,
								num_threads=config['num_threads'],
								batch_size=config['batch_size'],
								device=device,
								**dali_params['tensorflow_args'],
							)
						# default case if other dali loaders get added
						else:
							data_loader = data_loader_lut[data_loader_name](
								pipeline=dali_params['pipeline'],
								pipe_args=pipe_args,
								num_threads=config['num_threads'],
								batch_size=config['batch_size'],
								device=device,
							)

					# function being benchmarked, actual data set consumption
					def benchmark_func(
						base_data_source,
						data_loader_lut,
						data_loader_name,
						conversion_path,
						dali_params,
						config,
						device,
					):
						global data_loader
						data_loader()

					# surround in try except so in case of error other runs can still execute
					try:
						benchmark.add_run(
							f'{data_format} {data_loader_name} {device}',
							benchmark_func,
							setup=setup_func,
							prerun=prerun_func,
							rounds=args.rounds,
							kwargs={
								'base_data_source': base_data_source,
								'data_loader_lut': data_loader_lut,
								'data_loader_name': data_loader_name,
								'conversion_path': conversion_path,
								'dali_params': dali_params,
								'config': config,
								'device': device,
							},
						)
					except Exception as e:
						print(e)

		# if there where any successful runs, generate result graphs and save to results/<execution_name>
		print(benchmark.statistics())
		if len(benchmark.statistics()) > 0:
			benchmark.save_results()
			path = Path.cwd() / 'results' / execution_name_prefixed
			print(path.as_posix())
			path.mkdir(parents=True, exist_ok=True)

			result_data = {
				'system_information': benchmark.get_sysinfo(),
				'run_results': benchmark.statistics(),
				'used_diskspace': used_diskspace,
			}

			with open(path / 'raw_results.json', 'w') as result_file:
				json.dump(result_data, result_file, indent=4)

			Illustrator.illustrate_results(path, result_data, execution_name)

		# do clean up if necessary
		if args.delete_conversions:
			try:
				shutil.rmtree(output_path / execution_name, ignore_errors=True)
			except Exception:
				pass
