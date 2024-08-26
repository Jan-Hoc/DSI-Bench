import h5py
import zarr
import pickle
import numpy as np
import torch
import tensorflow as tf
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import os
from pathlib import Path
from subprocess import call


class DatasetNotAvailableError(Exception):
	pass


class ReturnTypeNotSupportedError(Exception):
	pass


class DataSourceBase:
	"""
	Base class for all data sources. This class is not meant to be used directly.

	Attributes
	----------
	path : Path | str
		the base path of the dataset (ignored for TEST data set)
	filename : str
		the filename of dataset file relative to 'path' (ignored for TEST data set)
	key : str | list
		the keys of the dataset which we consider (ignored for TEST data set)

	Methods
	-------
	__init__(self, path: Path | str, filename: str, key: list | str):
		creates DataSourceBase object
	check_data(self):
		not implemented for DataSourceBase
	__len__(self):
		not implemented for DataSourceBase
	__getitem__(self, index: int):
		not implemented for DataSourceBase
	get_collate_fn(self):
		collate function not required for TEST so returns None
	get_tf_output_signature(self):
		returns a tuple of tf.TensorSpec specifying format of samples
	get_tf_pad_types(self):
		padding not required for TEST so returns None
	set_return_type(self, return_type: str):
		sets return type to return_type for __getitem__ of child classes
	get_available_return_types(self):
		returns list of implemented return_types
	"""

	def __init__(self, path: None, filename: None, key: None):
		self.path = Path(path)
		self.filename = 'test.h5'
		self.filename = filename
		self.key = ['test']
		self.len = 10_000
		self.data = np.arange(self.len)
		self.return_type = 'python'

		self.check_data()

	def check_data(self):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError

	def __getitem__(self, index: int):
		raise NotImplementedError

	def get_collate_fn(self):
		return None

	def get_tf_output_signature(self):
		# sample is int
		return tf.TensorSpec((None), tf.int64)

	def get_tf_pad_types(self):
		return None

	def set_return_type(self, return_type: str):
		if return_type not in self.get_available_return_types():
			raise ReturnTypeNotSupportedError(f'Return type {return_type} not supported')

		self.return_type = return_type

	def get_available_return_types(self):
		return_types = [
			'python',  # return native python
			'tensorflow',  # return tensorflow tensor
			'pytorch',  # return pytorch tensor
		]

		return return_types


class DataSource(DataSourceBase):
	"""
	The initial dataset class. This class contains functions to load the data using it's original storage format.
	In this case, we are loading directly from np array

	Attributes
	----------
	path : Path | str
		the base path of the dataset (ignored for TEST data set)
	filename : str
		the filename of dataset file relative to 'path' (ignored for TEST data set)
	key : str | list
		the keys of the dataset which we consider (ignored for TEST data set)

	Methods
	-------
	__init__(self, path: Path | str, filename: str, key: list | str):
		creates DataSource object
	check_data(self):
		not needed for raw format of TEST data set
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		always acts as if return_type == python regardless of actual value
	"""

	def __init__(self, path: None = None, filename: None = None, key: None = None):
		super().__init__(path, filename, key)

	def prepare(self):
		pass

	def check_data(self):
		pass

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		sample = self.data[idx]

		return tuple([sample])


#################################
####### RAW STORAGE FORMAT ######
#################################


class DataSourceRaw(DataSource):
	"""
	This class contains functions to load the data using it's original (raw) storage format.
	In this case, we are loading directly from np array

	Attributes
	----------
	path : Path | str
		the base path of the dataset (ignored for TEST data set)
	filename : str
		the filename of dataset file relative to 'path' (ignored for TEST data set)
	key : str | list
		the keys of the dataset which we consider (ignored for TEST data set)

	Methods
	-------
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		adapts content of tuple depending on set return type
	"""

	def __getitem__(self, index: int):
		idx = index % self.len

		sample = self.data[idx]

		# convert features of sample to according return type
		if self.return_type == 'pytorch':
			return torch.Tensor([sample])
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor([sample], dtype=tf.int64)
		else:
			return tuple([sample])


def convert_to_raw(source: DataSource, output_path: Path | str) -> DataSourceRaw:
	"""
	convert DataSource to format raw and return according DataSourceRaw object
	for TEST data set we load directly out of memory for raw, so nothing to save

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion (ignored for raw format in this data set)

	Returns:
		DataSourceRaw: reads data in raw format from output_path
	"""

	output_path = Path(output_path)
	target_filename = Path(source.filename).stem + '.h5'

	return DataSourceRaw(output_path, target_filename, source.key)


#################################
###### HDF5 STORAGE FORMAT ######
#################################


class DataSourceHDF5(DataSource):
	"""
	This class contains functions to load the data using the hdf5 storage format.
	This is equivalent to the raw format

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	filename : str
		the filename of the hdf5 file of this dataset relative to `path`
	key : str | list
		the keys of the data set which we consider (ignored for TEST data set)

	Methods
	-------
	check_data(self):
		set len of data set based on amount of samples in hdf5 file
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		adapts content of tuple depending on set return type
	"""

	def __init__(self, path: Path | str, filename: str, key: list | str):
		super().__init__(path, filename, key)

	def check_data(self):
		try:
			with h5py.File(self.path / self.filename, 'r') as fh:
				self.len = fh[self.key[0]].shape[0]
		except Exception:
			raise DatasetNotAvailableError('Failed to load dataset')

	def __getitem__(self, index: int):
		idx = index % self.len

		with h5py.File(self.path / self.filename, 'r') as f:
			sample = int(f[self.key[0]][idx, ...])

		# convert features of sample based on return type
		if self.return_type == 'pytorch':
			return torch.Tensor([sample])
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(sample, dtype=tf.int64)
		else:
			return tuple([sample])


def convert_to_hdf5(source: DataSource, output_path: Path | str) -> DataSourceHDF5:
	"""
	convert DataSource to format hdf5 and return according DataSourceHDF5 object

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Raises:
		RuntimeError: raised if we try to overwrite original

	Returns:
		DataSourceHDF5: reads data in hdf5 format from output_path
	"""

	output_path = Path(output_path)
	target_filename = Path(source.filename).stem + '.h5'

	try:
		return DataSourceHDF5(output_path, target_filename, source.key)
	except DatasetNotAvailableError:
		print('converting to hdf5')

		os.makedirs(os.path.dirname(output_path / target_filename), exist_ok=True)

		with h5py.File(output_path / target_filename, 'w') as new_file:
			# copy data into hdf5 data set
			data = source.data
			new_file.create_dataset(
				data=data,
				name=source.key[0],
				dtype=np.float32,
				compression='gzip',
				compression_opts=7,
			)

		return DataSourceHDF5(output_path, target_filename, source.key)


#################################
###### ZARR STORAGE FORMAT ######
#################################


class DataSourceZarr(DataSourceBase):
	"""
	This class contains functions to load the data using the zarr storage format.

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	filename : str
		the filename of the zarr file of this dataset relative to `path`
	key : str | list
		the keys of the data set which we consider (ignored for TEST data set)

	Methods
	-------
	check_data(self):
		check if file present and set len of data set based on amount of samples in zarr file
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		adapts content of tuple depending on set return type
	"""

	def __init__(self, path: Path | str, filename: str, key: list | str):
		super().__init__(path, filename, key)

	def check_data(self):
		try:
			data = zarr.open_group(self.path / self.filename, mode='r')
			# set length and check our key exists
			self.len = data[self.key[0]].shape[0]
		except zarr.errors.GroupNotFoundError:
			raise DatasetNotAvailableError
		return

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# read features out of zarr file (only 1 for TEST data set)
		data = zarr.open_group(self.path / self.filename, mode='r')
		sample = []
		for k in self.key:
			sample.append(data[k][idx])

		# convert sample based on return type
		if self.return_type == 'pytorch':
			return torch.Tensor(sample)
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(sample, dtype=tf.int64)
		else:
			return tuple(sample)


def convert_to_zarr(source: DataSource, output_path: Path | str) -> DataSourceZarr:
	"""
	convert DataSource to format zarr and return according DataSourceZarr object

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceZarr: reads data in zarr format from output_path
	"""

	output_path = Path(output_path)
	target_filename = Path(source.filename).stem + '.zarr'

	try:
		return DataSourceZarr(output_path, target_filename, source.key)
	except DatasetNotAvailableError:
		print('converting to zarr')

		# copy required data into zarr group
		dest = zarr.open_group(output_path / target_filename, mode='w')
		dest[source.key[0]] = source.data

		return DataSourceZarr(output_path, target_filename, source.key)


#################################
##### PICKLE STORAGE FORMAT #####
#################################


class DataSourcePickle(DataSourceBase):
	"""
	This class contains functions to load the data using the pickle storage format.

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	filename : str
		the DIRECTORY containing pickle files of samples from this dataset, relative to `path`
	key : str | list
		the keys of the data set which we consider (ignored for TEST data set)

	Methods
	-------
	check_data(self):
		sets len based on amount of sample files and checks if first sample contains required amount of keys
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		adapts content of tuple depending on set return type
	"""

	def __init__(self, path: Path | str, filename: str, key: list | str):
		super().__init__(path, filename, key)

	def check_data(self):
		if not (self.path / self.filename).exists():
			raise DatasetNotAvailableError

		self.len = len(list((self.path / self.filename).glob('*.pickle')))

		sample_data = self[0]
		if len(sample_data) != len(self.key):
			print(
				f'Warning: sample data has {len(sample_data)} keys, but {len(self.key)} keys were specified. '
			)
			raise DatasetNotAvailableError

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# read pickle file corresponding to sample <index>
		with open(self.path / self.filename / f'{idx}.pickle', 'rb') as f:
			sample = pickle.load(f)

		# convert sample based on return type
		if self.return_type == 'pytorch':
			return torch.Tensor(sample)
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(sample, dtype=tf.int64)
		else:
			return tuple(sample)


def convert_to_pickle(source: DataSource, output_path: Path | str) -> DataSourcePickle:
	"""
	convert DataSource to format pickle and return according DataSourcePickle object
	Every sample is saved in individual pickle file

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourcePickle: reads data in pickle format from output_path
	"""

	output_path = Path(output_path)
	target_filename = Path(Path(source.filename).stem + '.picklestore')

	try:
		return DataSourcePickle(output_path, target_filename, source.key)
	except DatasetNotAvailableError:
		print('converting to pickle')

		# convert all samples to individual pickle files
		os.makedirs(output_path / target_filename, exist_ok=True)
		for i in range(len(source)):
			with open(output_path / target_filename / f'{i}.pickle', 'wb') as f:
				# pickle the 'data' dictionary using the highest protocol available.
				pickle.dump(source[i], f, pickle.HIGHEST_PROTOCOL)

		return DataSourcePickle(output_path, target_filename, source.key)


################################
###### NPY STORAGE FORMAT ######
################################


class DataSourceNPY(DataSourceBase):
	"""
	This class contains functions to load the data using the npy storage format.

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	filename : str
		the DIRECTORY containing npy files of samples from this dataset, relative to `path`
	key : str | list
		the keys of the data set which we consider (ignored for TEST data set)

	Methods
	-------
	check_data(self):
		sets len and checks if first sample contains all required keys
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		adapts content of tuple depending on set return type
	"""

	def __init__(self, path: Path | str, filename: str, key: list | str):
		super().__init__(path, filename, key)

	def check_data(self):
		if not (self.path / self.filename / self.key[0]).exists():
			raise DatasetNotAvailableError

		self.len = len(list((self.path / self.filename / self.key[0]).glob('*.npy')))

		sample_data = self[0]
		if len(sample_data) != len(self.key):
			print(
				f'Warning: sample data has {len(sample_data)} keys, but {len(self.key)} keys were specified. '
			)
			raise DatasetNotAvailableError

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		sample = []
		# gather data from individual feature npy files
		for k in self.key:
			sample.append(np.load(self.path / self.filename / k / f'{idx}.npy'))

		# convert sample to according return type
		if self.return_type == 'pytorch':
			return torch.from_numpy(np.array(sample, copy=False))
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(sample, dtype=tf.int64)
		else:
			return tuple(sample)


def convert_to_npy(source: DataSource, output_path: Path | str) -> DataSourceNPY:
	"""
	convert DataSource to format npy and return according DataSourceNPY object
	every feature of every sample is saved in an individual file and grouped into a subfolder by feature
	individual files for features needed due to different dimensions

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceNPY: reads data in npy format from output_path
	"""

	output_path = Path(output_path)
	target_filename = Path(Path(source.filename).stem + '.npy')

	try:
		return DataSourceNPY(output_path, target_filename, source.key)
	except DatasetNotAvailableError:
		print('converting to npy')

		os.makedirs(output_path / target_filename, exist_ok=True)
		# create directory for every key
		for k in source.key:
			os.makedirs(output_path / target_filename / k, exist_ok=True)

		# convert samples to npy files for every feature and save to according directory
		for i in range(len(source)):
			sample = source[i]

			for k in source.key:
				np.save(output_path / target_filename / k / f'{i}.npy', sample[source.key.index(k)])

		return DataSourceNPY(output_path, target_filename, source.key)


################################
###### NPZ STORAGE FORMAT ######
################################


class DataSourceNPZ(DataSourceBase):
	"""
	This class contains functions to load the data using the npz storage format.

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	filename : str
		the DIRECTORY containing npz files of samples from this dataset, relative to `path`
	key : str | list
		the keys of the data set which we consider (ignored for TEST data set)

	Methods
	-------
	check_data(self):
		sets len and checks if first sample contains required amount of keys
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		adapts content of tuple depending on set return type
	"""

	def __init__(self, path: Path | str, filename: str, key: list | str):
		super().__init__(path, filename, key)

	def check_data(self):
		if not (self.path / self.filename).exists():
			raise DatasetNotAvailableError
		self.len = len(list((self.path / self.filename).glob('*.npz')))

		sample_data = self[0]
		if len(sample_data) != len(self.key):
			print(
				f'Warning: sample data has {len(sample_data)} keys, but {len(self.key)} keys were specified. '
			)
			raise DatasetNotAvailableError

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# read sample from npz file
		data = np.load(self.path / self.filename / f'{idx}.npz')

		sample = []
		for k in self.key:
			# make sure type is correct
			sample.append(data[k].astype(np.int64, copy=False))

		# convert sample according to return type
		if self.return_type == 'pytorch':
			return torch.from_numpy(np.array(sample, copy=False))
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(sample, dtype=tf.int64)
		else:
			return tuple(sample)


def convert_to_compressed_npz(source: DataSource, output_path: Path | str) -> DataSourceNPZ:
	"""
	convert DataSource to format npz and return according DataSourceNPZ object
	every sample is saved in individual file

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceNPZ: reads data in npz format from output_path
	"""

	output_path = Path(output_path)
	target_filename = Path(Path(source.filename).stem + '.npz')

	try:
		return DataSourceNPZ(output_path, target_filename, source.key)
	except DatasetNotAvailableError:
		print('converting to compressed npz')

		os.makedirs(output_path / target_filename, exist_ok=True)

		# convert samples to individual npz files

		for i in range(len(source)):
			sample = source[i]
			sample_d = {}
			for k in source.key:
				sample_d[k] = sample[source.key.index(k)]
			np.savez_compressed(output_path / target_filename / f'{i}.npz', **sample_d)

		return DataSourceNPZ(output_path, target_filename, source.key)


######################################
### TFRECORD_SINGLE STORAGE FORMAT ###
######################################


class DataSourceTFRecordSingle(DataSourceBase):
	"""
	This class contains functions to load the data using the tfrecord_single storage format.

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	filename : str
		the filename of the tfrecord file containing the samples from this dataset, relative to `path`
	key : str | list
		the keys of the data set which we consider (ignored for TEST data set)

	Methods
	-------
	check_data(self):
		sets len and checks if first sample contains all required keys
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		adapts content of tuple depending on set return type
	"""

	def __init__(self, path: Path | str, filename: str, key: list | str):
		# set device to CPU to avoid weird errors in GPU environment during data loading
		self.device = tf.config.list_logical_devices('CPU')[0].name
		super().__init__(path, filename, key)

	def check_data(self):
		if not (self.path / self.filename).exists():
			raise DatasetNotAvailableError

		with tf.device(self.device):
			self.dataset = tf.data.TFRecordDataset(
				self.path / self.filename, num_parallel_reads=tf.data.AUTOTUNE
			)
			self.len = sum(1 for _ in self.dataset)  # calculate length of data set

			sample_data = self[0]
			if len(sample_data) != len(self.key):
				print(
					f'Warning: sample data has {len(sample_data)} keys, but {len(self.key)} keys were specified. '
				)
				raise DatasetNotAvailableError

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		with tf.device(self.device):
			# skip beginning of data set to retrieve sample <index>
			elem = None
			for data in self.dataset.skip(idx).take(1):
				elem = data

			sample = []

			# create parsing dictionary to get serialized tensors for all features
			parse_dic = {}
			for k in self.key:
				parse_dic[k] = tf.io.FixedLenFeature([], tf.string)
			serialized_bytes = tf.io.parse_single_example(elem, parse_dic)

			# deserialize features and convert to corresponding output type
			if self.return_type == 'pytorch':
				for k in self.key:
					feature = serialized_bytes[k]
					sample.append(tf.io.parse_tensor(feature, out_type=tf.int64).numpy())
				return torch.from_numpy(np.array(sample, copy=False))
			elif self.return_type == 'tensorflow':
				for k in self.key:
					feature = serialized_bytes[k]
					sample.append(tf.io.parse_tensor(feature, out_type=tf.int64))
				return tf.convert_to_tensor(sample)
			else:
				for k in self.key:
					feature = serialized_bytes[k]
					sample.append(tf.io.parse_tensor(feature, out_type=tf.int64).numpy())
				return tuple(sample)


def convert_to_tfrecords_single(
	source: DataSource, output_path: Path | str
) -> DataSourceTFRecordSingle:
	"""
	convert DataSource to format tfrecord_single and return according DataSourceTFRecordSingle object
	all samples are saved in one single TFRecord file

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceTFRecordSingle: reads data in tfrecord_single format from output_path
	"""

	output_path = Path(output_path)
	target_filename = Path(Path(source.filename).stem + '.tfrecords')

	try:
		return DataSourceTFRecordSingle(output_path, target_filename, source.key)
	except DatasetNotAvailableError:
		print('converting to tfrecord single')

		os.makedirs(output_path, exist_ok=True)
		fname_out = Path(output_path / target_filename).absolute().as_posix()

		with tf.io.TFRecordWriter(fname_out) as fi:
			for i in range(len(source)):
				sample = source[i]
				sample_d = {}

				# serialize individual features to avoid issues with dimensions
				for k_i, k in enumerate(source.key):
					sample_d[k] = tf.train.Feature(
						bytes_list=tf.train.BytesList(
							value=[tf.io.serialize_tensor(sample[k_i]).numpy()]
						)
					)

				# save sample to file
				record_bytes = tf.train.Example(
					features=tf.train.Features(feature=sample_d)
				).SerializeToString()
				fi.write(record_bytes)

		return DataSourceTFRecordSingle(output_path, target_filename, source.key)


#####################################
### TFRECORD_MULTI STORAGE FORMAT ###
#####################################


class DataSourceTFRecordMulti(DataSourceBase):
	"""
	This class contains functions to load the data using the tfrecord_multi storage format.

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	filename : str
		the DIRECTORY containing the tfrecord files of the samples from this dataset, relative to `path`
	key : str | list
		the keys of the data set which we consider (ignored for TEST data set)

	Methods
	-------
	check_data(self):
		sets len and checks if first sample contains all required keys
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns the values of specified keys from sample <index> as a tuple
		adapts content of tuple depending on set return type
	"""

	def __init__(self, path: Path | str, filename: str, key: list | str):
		self.device = tf.config.list_logical_devices('CPU')[0].name
		super().__init__(path, filename, key)

	def check_data(self):
		if not (self.path / self.filename).exists():
			raise DatasetNotAvailableError

		self.len = len(list((self.path / self.filename).glob('*.tfrecord')))

		sample_data = self[0]
		if len(sample_data) != len(self.key):
			print(
				f'Warning: sample data has {len(sample_data)} keys, but {len(self.key)} keys were specified. '
			)
			raise DatasetNotAvailableError

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		with tf.device(self.device):
			# read according sample
			dataset = tf.data.TFRecordDataset(
				self.path / self.filename / f'{idx}.tfrecord', num_parallel_reads=tf.data.AUTOTUNE
			)

			elem = None
			for data in dataset:
				elem = data

			sample = []

			# create parsing dictionary to get serialized tensors for all features
			parse_dic = {}
			for k in self.key:
				parse_dic[k] = tf.io.FixedLenFeature([], tf.string)
			serialized_bytes = tf.io.parse_single_example(elem, parse_dic)

			# deserialize features and convert to corresponding return type
			if self.return_type == 'pytorch':
				for k in self.key:
					feature = serialized_bytes[k]
					sample.append(tf.io.parse_tensor(feature, out_type=tf.int64).numpy())
				return torch.from_numpy(np.array(sample, copy=False))
			elif self.return_type == 'tensorflow':
				for k in self.key:
					feature = serialized_bytes[k]
					sample.append(tf.io.parse_tensor(feature, out_type=tf.int64))
				return tf.convert_to_tensor(sample)
			else:
				for k in self.key:
					feature = serialized_bytes[k]
					sample.append(tf.io.parse_tensor(feature, out_type=tf.int64).numpy())
				return tuple(sample)


def convert_to_tfrecords_multi(
	source: DataSource, output_path: Path | str
) -> DataSourceTFRecordMulti:
	"""
	convert DataSource to format tfrecord_multi and return according DataSourceTFRecordMulti object
	all samples are saved in individual TFRecord files

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceTFRecordMulti: reads data in tfrecord_multi format from output_path
	"""

	output_path = Path(output_path)
	target_filename = Path(Path(source.filename).stem + '.tfrecords')  # it's actually a dir

	try:
		return DataSourceTFRecordMulti(output_path, target_filename, source.key)
	except DatasetNotAvailableError:
		print('converting to tfrecord multi')

		os.makedirs(output_path / target_filename, exist_ok=True)

		# convert individual samples to tfrecord files
		for i in range(len(source)):
			fname_out = Path(output_path / target_filename / f'{i}.tfrecord').absolute().as_posix()
			sample = source[i]
			sample_d = {}
			for k_i, k in enumerate(source.key):
				sample_d[k] = tf.train.Feature(
					bytes_list=tf.train.BytesList(
						value=[tf.io.serialize_tensor(sample[k_i]).numpy()]
					)
				)

			# save sample individually
			with tf.io.TFRecordWriter(fname_out) as fi:
				record_bytes = tf.train.Example(
					features=tf.train.Features(feature=sample_d)
				).SerializeToString()
				fi.write(record_bytes)

		return DataSourceTFRecordMulti(output_path, target_filename, source.key)


###################################
#### AVAILABLE STORAGE FORMATS ####
###################################


def get_available_converters():
	formats = {
		'raw': convert_to_raw,
		'hdf5': convert_to_hdf5,
		'zarr': convert_to_zarr,
		'pickle': convert_to_pickle,
		'npy': convert_to_npy,
		'npz': convert_to_compressed_npz,
		'tfrecord_single': convert_to_tfrecords_single,
		'tfrecord_multi': convert_to_tfrecords_multi,
	}

	return formats


###############################
###### NPY DALI PIPELINE ######
###############################


def dali_npy_prepare(source: DataSource, output_path: str | Path) -> tuple:
	"""
	make sure all conversions are done and return arguments required for npy pipeline

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		tuple: tuple containing arguments for npy pipeline
	"""

	# make sure data is converted
	npy = get_available_converters()['npy'](source, output_path)

	# directory containing npy files
	file_path = npy.path / npy.filename

	# list of all relevant npy files
	file_list = [str(i) + '.npy' for i in range(len(npy))]

	return (file_path, file_list, npy.key)


@pipeline_def
def dali_npy_pipeline(file_path: str | Path, file_list: list, keys: list, device: str):
	"""
	DALI pipeline definition using numpy reader

	Args:
		file_path (str | Path): root directory of npy subfolders
		file_list (list): list of numpy files in subfolders
		keys (list): relevant keys of features
		device (str): device to load samples to
	"""
	sample = []

	# read individual features from different directories
	for k in keys:
		sample.append(
			fn.readers.numpy(
				device=device,
				file_root=file_path / k,
				files=file_list,
				name='reader' + k,
				random_shuffle=True,
				seed=193,
				lazy_init=True,
			)
		)

	return tuple(sample)


#####################################
### TFRECORD_SINGLE DALI PIPELINE ###
#####################################


def dali_tfrecord_single_prepare(source: DataSource, output_path: str | Path) -> tuple:
	"""
	make sure all conversions are done and return arguments required for tfrecord_single pipeline

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		tuple: tuple containing arguments for tfrecord_single pipeline
	"""

	# make sure conversion is done
	tf_single = get_available_converters()['tfrecord_single'](source, output_path)

	# path of tfrecord file
	file_path = tf_single.path / tf_single.filename

	# create index file required by tfrecord reader
	idx_path = tf_single.path / 'idx_files'
	if not idx_path.is_dir():
		idx_path.mkdir(parents=True, exist_ok=True)

	idx_path = idx_path / 'test.idx'
	if not idx_path.is_file():
		call(['tfrecord2idx', file_path, idx_path])

	return (file_path, idx_path, source.key)


@pipeline_def
def dali_tfrecord_single_pipeline(
	file_path: str | Path, idx_path: str | Path, keys: list, device: str
):
	"""
	DALI pipeline definition using tfrecord reader

	Args:
		file_path (str | Path): path to tfrecord file
		idx_path (str | Path): path to index file
		keys (list): relevant keys of features
		device (str): device to load samples to
	"""

	# define what type of features we have
	# string since tensors are serialized
	features = {}
	for k in keys:
		features[k] = tfrec.FixedLenFeature([], tfrec.string, '')

	# read different features out of tfrecord file for a sample
	inputs = fn.readers.tfrecord(
		path=file_path,
		index_path=idx_path,
		features=features,
		random_shuffle=True,
		seed=193,
		lazy_init=True,
		name='reader' + keys[0],
	)

	data = []
	for k in keys:
		data.append(inputs[k])

	return tuple(data)


##################################
#### AVAILABLE DALI PIPELINES ####
##################################


def get_available_pipelines(keys: list):
	pipelines = {
		'npy': {
			'preparation': dali_npy_prepare,
			'pipeline': dali_npy_pipeline,
			'labels': keys,
			'pytorch_args': {},
			'tensorflow_args': {'shapes': [() for _ in keys], 'dtypes': [tf.int64 for _ in keys]},
		},
		'tfrecord_single': {
			'preparation': dali_tfrecord_single_prepare,
			'pipeline': dali_tfrecord_single_pipeline,
			'labels': keys,
			'pytorch_args': {},
			'tensorflow_args': {'shapes': [() for _ in keys], 'dtypes': [tf.int64 for _ in keys]},
		},
	}

	return pipelines
