import yaml
import os


def parse_configs() -> dict:
	"""
	Parse the different configuration files for the benchmark suites in configs/

	Returns:
		dict: dict containing a mapping suite_name -> config content
	"""
	configs = {}
	config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')

	for file_name in os.listdir(config_path):
		if file_name.endswith('.yaml') or file_name.endswith('.yml'):
			file_path = os.path.join(config_path, file_name)
			with open(file_path, 'r') as config_file:
				config_data = yaml.safe_load(config_file)
				if ('dataset_type' in config_data) and ('suite_name' in config_data):
					configs[config_data['suite_name']] = config_data

	return configs
