import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm


class Illustrator:
	"""
	Class to create graphs for results of a benchmark execution

	Methods
	-------
	illustrate_results(path: str | Path, result_data: dict, execution_name: str)
		create graphs to summarize result of execution, compare data loaders and to compare storage formats
	"""

	# meta data about metrics for graph creation
	__metrics = {
		'cpu': {'ylabel': 'Percent', 'factor': 1, 'title': 'CPU Utilization of Data Set'},
		'ram': {'ylabel': 'MiB', 'factor': 1 / (1024**2), 'title': 'RAM Utilization of Data Set'},
		'time': {'ylabel': 'Seconds', 'factor': 1, 'title': 'Runtime of Data Set'},
	}
	# color map for summary graphs
	__color_map = LinearSegmentedColormap.from_list(
		'green_to_red', [(0, 'green'), (0.15, 'forestgreen'), (0.6, 'orange'), (1, 'red')]
	)

	def illustrate_results(path: str | Path, result_data: dict, execution_name: str) -> None:
		"""
		create illustrative graphs from benchmark results

		Args:
			path (str | Path): path where to save result graphs
			result_data (dict): dict of result data as saved to json by benchmarks.py
			execution_name (str): name of benchmark execution for which graphs are created
		"""
		path = Path(path)
		path.mkdir(parents=True, exist_ok=True)

		# create graph to compare used disk space by different conversions
		Illustrator.__illustrate_used_diskspace(result_data['used_diskspace'], path, execution_name)
		# create graphs to compare data loaders
		Illustrator.__illustrate_data_loaders(
			result_data['run_results'], path / 'data_loader_graphs', execution_name
		)
		# create graphs to compare converters
		Illustrator.__illustrate_converters(
			result_data['run_results'], path / 'converter_graphs', execution_name
		)
		# create heat maps to summarize results of runs
		Illustrator.__illustrate_summary_graphs(
			result_data['run_results'], path / 'summary_graphs', execution_name
		)

	def __illustrate_used_diskspace(used_diskspace: dict, path: Path, execution_name: str) -> None:
		"""
		create a graph to illustrate used disk space by conversions of dataset

		Args:
			used_diskspace (dict): dict containing mapping converter_name => used space in MiB
			path (Path): path where to save image
			execution_name (str): name of dataset
		"""
		path = path / f'{execution_name}_diskspace.png'

		fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
		bar_plot = ax.bar(used_diskspace.keys(), used_diskspace.values())
		ax.bar_label(bar_plot, fmt=lambda x: f'{round(x, 2)}')
		ax.set(
			title=f'Used Disk Space by converted Data Set {execution_name} (MiB)',
			ylabel='MiB',
		)
		plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
		ax.margins(y=0.1)

		fig.tight_layout()
		fig.savefig(path, dpi=300, bbox_inches='tight')

		plt.close('all')

	def __illustrate_data_loaders(
		run_results: dict,
		path: Path,
		execution_name: str,
	) -> None:
		"""
		create graphs for every converter to compare data loaders

		Args:
			used_diskspace (dict): dict containing run_results
			path (Path): path where to save image
			execution_name (str): name of dataset
		"""
		# check which out types where already seen to account for dali where not specified
		seen_out_types = []
		for run_name in run_results:
			elems = run_name.split(' ')
			if len(elems) == 4:
				if elems[3] not in seen_out_types:
					seen_out_types.append(elems[3])
		if len(seen_out_types) == 0:
			seen_out_types = ['default']

		# gather data in map
		result_map = {}
		for run_name, run_data in run_results.items():
			elems = run_name.split(' ')
			if len(elems) == 4:
				for metric in Illustrator.__metrics:
					Illustrator.__create_sub_dicts(
						result_map, [elems[0], metric, elems[2], elems[3]]
					)
					result_map[elems[0]][metric][elems[2]][elems[3]][elems[1]] = {
						'mean': run_data[metric]['total']['mean'],
						'stddev': run_data[metric]['total']['stddev'],
					}
			else:  # case dali loader
				for out_type in seen_out_types:
					for metric in Illustrator.__metrics:
						Illustrator.__create_sub_dicts(
							result_map, [elems[0], metric, elems[2], out_type]
						)
						result_map[elems[0]][metric][elems[2]][out_type][elems[1]] = {
							'mean': run_data[metric]['total']['mean'],
							'stddev': run_data[metric]['total']['stddev'],
						}

		# generate actual graphs
		for converter_name, converter_data in result_map.items():
			for metric, metric_data in converter_data.items():
				(path / metric).mkdir(parents=True, exist_ok=True)
				for device, device_data in metric_data.items():
					for out_type, out_type_data in device_data.items():
						means = []
						errors = []
						labels = []
						for loader, data in out_type_data.items():
							means.append(data['mean'])
							errors.append(data['stddev'])
							labels.append(loader)

						fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
						bar_plot = ax.bar(
							labels,
							np.array(means) * Illustrator.__metrics[metric]['factor'],
							yerr=np.array(errors) * Illustrator.__metrics[metric]['factor'],
						)
						ax.bar_label(bar_plot, fmt=lambda x: f'{round(x, 2)}')
						ax.set(
							title=f'{Illustrator.__metrics[metric]["title"]} {execution_name} for storage format {converter_name}',
							ylabel=Illustrator.__metrics[metric]['ylabel'],
						)
						plt.setp(
							ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor'
						)
						ax.margins(y=0.1)

						fig.tight_layout()
						fig.savefig(
							path / metric / f'{converter_name}_{metric}_{device}_{out_type}.png',
							dpi=300,
							bbox_inches='tight',
						)

						plt.close('all')

	def __illustrate_converters(
		run_results: dict,
		path: Path,
		execution_name: str,
	) -> None:
		"""
		create graphs for every data loader to compare converter types

		Args:
			used_diskspace (dict): dict containing run_results
			path (Path): path where to save image
			execution_name (str): name of dataset
		"""
		# check which out types where already seen to account for dali where not specified
		seen_out_types = []
		for run_name in run_results:
			elems = run_name.split(' ')
			if len(elems) == 4:
				if elems[3] not in seen_out_types:
					seen_out_types.append(elems[3])
		if len(seen_out_types) == 0:
			seen_out_types = ['default']

		# gather data in map
		result_map = {}
		for run_name, run_data in run_results.items():
			elems = run_name.split(' ')
			if len(elems) == 4:
				for metric in Illustrator.__metrics:
					Illustrator.__create_sub_dicts(
						result_map, [elems[1], metric, elems[2], elems[3]]
					)
					result_map[elems[1]][metric][elems[2]][elems[3]][elems[0]] = {
						'mean': run_data[metric]['total']['mean'],
						'stddev': run_data[metric]['total']['stddev'],
					}
			else:  # case dali loader
				for out_type in seen_out_types:
					for metric in Illustrator.__metrics:
						Illustrator.__create_sub_dicts(
							result_map, [elems[1], metric, elems[2], out_type]
						)
						result_map[elems[1]][metric][elems[2]][out_type][elems[0]] = {
							'mean': run_data[metric]['total']['mean'],
							'stddev': run_data[metric]['total']['stddev'],
						}

		# generate actual graphs
		for loader_name, loader_data in result_map.items():
			for metric, metric_data in loader_data.items():
				(path / metric).mkdir(parents=True, exist_ok=True)
				for device, device_data in metric_data.items():
					for out_type, out_type_data in device_data.items():
						means = []
						errors = []
						labels = []
						for converter, data in out_type_data.items():
							means.append(data['mean'])
							errors.append(data['stddev'])
							labels.append(converter)

						fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
						bar_plot = ax.bar(
							labels,
							np.array(means) * Illustrator.__metrics[metric]['factor'],
							yerr=np.array(errors) * Illustrator.__metrics[metric]['factor'],
						)
						ax.bar_label(bar_plot, fmt=lambda x: f'{round(x, 2)}')
						ax.set(
							title=f'{Illustrator.__metrics[metric]["title"]} {execution_name} for consumer {loader_name}',
							ylabel=Illustrator.__metrics[metric]['ylabel'],
						)
						plt.setp(
							ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor'
						)
						ax.margins(y=0.1)

						fig.tight_layout()
						fig.savefig(
							path / metric / f'{loader_name}_{metric}_{device}_{out_type}.png',
							dpi=300,
							bbox_inches='tight',
						)

						plt.close('all')

	def __illustrate_summary_graphs(
		run_results: dict,
		path: Path,
		execution_name: str,
	) -> None:
		"""
		create comprehensive graphs for easy comparison

		Args:
			used_diskspace (dict): dict containing run_results
			path (Path): path where to save image
			execution_name (str): name of dataset
		"""
		# check which out types where already seen to account for dali where not specified
		seen_out_types = []
		seen_devices = []
		seen_converters = []
		seen_data_loaders = []
		for run_name in run_results:
			elems = run_name.split(' ')
			if elems[0] not in seen_converters:
				seen_converters.append(elems[0])
			if elems[1] not in seen_data_loaders:
				seen_data_loaders.append(elems[1])
			if elems[2] not in seen_devices:
				seen_devices.append(elems[2])
			if len(elems) == 4:
				if elems[3] not in seen_out_types:
					seen_out_types.append(elems[3])
		if len(seen_out_types) == 0:
			seen_out_types = ['default']

		# gather data in map
		result_map = {}
		for metric in Illustrator.__metrics:
			result_map[metric] = {}
			for device in seen_devices:
				result_map[metric][device] = {}
				for out_type in seen_out_types:
					result_map[metric][device][out_type] = {}
					for loader in seen_data_loaders:
						result_map[metric][device][out_type][loader] = {}
						for converter in seen_converters:
							result_map[metric][device][out_type][loader][converter] = {
								'mean': 0,
								'stddev': 0,
							}

		for run_name, run_data in run_results.items():
			elems = run_name.split(' ')
			if len(elems) == 4:
				for metric in Illustrator.__metrics:
					result_map[metric][elems[2]][elems[3]][elems[1]][elems[0]] = {
						'mean': run_data[metric]['total']['mean'],
						'stddev': run_data[metric]['total']['stddev'],
					}
			else:  # case dali loader
				for out_type in seen_out_types:
					for metric in Illustrator.__metrics:
						result_map[metric][elems[2]][out_type][elems[1]][elems[0]] = {
							'mean': run_data[metric]['total']['mean'],
							'stddev': run_data[metric]['total']['stddev'],
						}

		# generate actual graphs
		for metric, metric_data in result_map.items():
			(path / metric).mkdir(parents=True, exist_ok=True)
			for device, device_data in metric_data.items():
				for out_type, out_type_data in device_data.items():
					# calculate values, std (errors) and where no data is available
					values = []
					errors = []
					for loader_name, loader_values in out_type_data.items():
						data = list(loader_values.values())
						values.append(list(map(lambda x: x['mean'], data)))
						errors.append(list(map(lambda x: x['stddev'], data)))

					values = np.array(values) * Illustrator.__metrics[metric]['factor']
					values = np.ma.masked_where(values == 0, values)
					errors = np.array(errors) * Illustrator.__metrics[metric]['factor']
					errors = np.ma.masked_where(errors == 0, errors)

					value_mask = np.ma.getmaskarray(values)
					error_mask = np.ma.getmaskarray(errors)

					minVal = np.min(values[np.nonzero(values)])
					maxVal = np.max(values[np.nonzero(values)])

					# version without labels
					fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
					im = ax.imshow(
						values,
						cmap=Illustrator.__color_map,
						interpolation='nearest',
						norm=LogNorm(vmin=minVal, vmax=maxVal),
					)

					cbar = fig.colorbar(im)
					cbar.ax.set_ylabel(
						Illustrator.__metrics[metric]['ylabel'],
						rotation=-90,
						va='bottom',
						fontsize='x-large',
					)

					# add min and max labels to colorbar
					cbar_min, cbar_max = im.get_clim()
					cbar.ax.text(
						0.5,
						-0.01,
						round(cbar_min, 2),
						color='black',
						transform=cbar.ax.transAxes,
						horizontalalignment='center',
						verticalalignment='top',
					)
					cbar.ax.text(
						0.5,
						1.01,
						round(cbar_max, 2),
						color='black',
						transform=cbar.ax.transAxes,
						horizontalalignment='center',
						verticalalignment='bottom',
					)

					ax.set_xticks(np.arange(len(seen_converters)), labels=seen_converters)
					ax.set_yticks(np.arange(len(seen_data_loaders)), labels=seen_data_loaders)

					plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

					for i in range(len(seen_data_loaders)):
						for j in range(len(seen_converters)):
							if value_mask[i][j] and error_mask[i][j]:
								ax.text(
									j,
									i,
									'N/A',
									ha='center',
									va='center',
									color='k',
									fontsize='small',
								)

					ax.set_title(
						f'Overview {Illustrator.__metrics[metric]["title"]} {execution_name}',
						fontsize='xx-large',
					)
					ax.set_xlabel('Storage Formats', fontsize='x-large')
					ax.set_ylabel('Consumers', fontsize='x-large')

					fig.tight_layout()
					fig.savefig(
						path / metric / f'{metric}_{device}_{out_type}.png',
						dpi=300,
						bbox_inches='tight',
					)

					# add version with labels in fields
					values = np.ma.getdata(values)
					errors = np.ma.getdata(errors)
					for i in range(len(seen_data_loaders)):
						for j in range(len(seen_converters)):
							if value_mask[i][j] and error_mask[i][j]:
								pass  # label already set previously for N/A case
							else:
								ax.text(
									j,
									i,
									f'{round(values[i][j], 2)}\u00b1{round(errors[i][j], 2)}',
									ha='center',
									va='center',
									color='w',
									fontsize='x-small',
								)

					fig.savefig(
						path / metric / f'{metric}_{device}_{out_type}_label.png',
						dpi=300,
						bbox_inches='tight',
					)

					plt.close('all')

	def __create_sub_dicts(d: dict, keys: list) -> None:
		"""
		helper to create nested dictionary

		Args:
			d (dict): dictionary
			keys (list): hierarchical list of keys
		"""
		for k in keys:
			if k not in d:
				d[k] = {}
			d = d[k]
