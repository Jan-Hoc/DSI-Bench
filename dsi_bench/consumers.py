from typing import Callable
import numpy as np
import concurrent.futures
import torch.utils.data
import tensorflow as tf
from tensorflow.python.keras.utils import all_utils
import nvidia.dali.plugin.tf as dali_tf
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy


class DataLoader:
	"""
	A class to load data of given data set

	Attributes
	----------
	dataset : DataSourceBase
	    data set containing data to be loaded
	prng : np.random.RandomState | None = None
		used to shuffle data if shuffle=True
	shuffle : bool = False
		if True data will be shuffled before iterating

	Methods
	-------
	__init__(self, dataset: 'DataSourceBase', prng: np.random.RandomState | None = None, shuffle: bool = False):
		creates DataLoader object
	__len__(self):
	    returns length of data set
	__iter__(self):
		(randomly) iterates through data set
	"""

	def __init__(
		self,
		dataset: 'DataSourceBase',
		prng: np.random.RandomState | None = None,
		shuffle: bool = False,
	):
		self.dataset = dataset

		# shuffle data if necessary
		if prng is None:
			prng = np.random.RandomState(193)
		self.inds = np.arange(len(self.dataset))
		if shuffle:
			prng.shuffle(self.inds)

	def __len__(self):
		return len(self.dataset)

	def __iter__(self):
		for i in self.inds:
			yield self.dataset[i]


#######################################
##### CLASSES FOR PYTHON CONSUMER #####
#######################################


class SampleMinibatch:
	"""
	Minibatch class for PythonConsumer

	Attributes
	----------
	dataset : DataSourceBase
	    data set containing data to be loaded
	num_threads : int
		amount of threads which should be used for loading
	batch_size : int
		batch size with for loading
	prng : np.random.RandomState | None = None
		used to shuffle data if shuffle=True
	shuffle : bool = False
		if True data will be shuffled before iterating

	Methods
	-------
	__init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, prng: np.random.RandomState | None = None, shuffle: bool = False, **kwargs):
		creates SampleMinibatch object
	__len__(self):
	    returns amount of batches in data set
	__getitem__(self, index: int):
		return batch index of data set
	"""

	def __init__(
		self,
		dataset: 'DataSourceBase',
		num_threads: int,
		batch_size: int,
		prng: np.random.RandomState | None = None,
		shuffle: int = False,
	):
		self.dataset = dataset
		self.num_threads = num_threads
		self.batch_size = batch_size
		self.num_samples = len(dataset)
		self.len = len(self)

		# shuffle data if necessary
		if prng is None:
			prng = np.random.RandomState(193)
		self.inds = np.arange(self.len * self.batch_size)
		if shuffle:
			prng.shuffle(self.inds)
		self.inds = np.reshape(self.inds, (self.len, self.batch_size))

	def __len__(self):
		return int(self.num_samples // self.batch_size)

	def worker_fn(self, ind: int):
		"""worker function loading element to pass to threads"""
		return self.dataset[ind]

	def __iter__(self):
		# load minibatches with num_threads threads
		# use threads instead of processes bc IO heavy and not CPU heavy workload
		with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
			for inds_ in self.inds:
				batch = executor.map(self.worker_fn, inds_)
				yield list(batch)


class PythonConsumer:
	"""
	Class using DataLoaderPython to consume entire data set

	Attributes
	----------
	dataset : DataSourceBase
	    data set containing data to be loaded
	num_threads : int
		amount of threads which should be used for loading
	batch_size : int
		batch size with for loading
	device : str
		is ignored but present for compatibility reasons with other consumers

	Methods
	-------
	__init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, prng: np.random.RandomState | None = None, shuffle: bool = False, **kwargs):
		creates PythonConsumer object using SampleMinibatch
	__call__(self):
		iterate through entire data set using created SampleMinibatch
	"""

	def __init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, device: str):
		self.dataloader = SampleMinibatch(
			dataset=dataset, num_threads=num_threads, batch_size=batch_size, shuffle=True
		)
		self.num_samples = len(self.dataloader)

	def __call__(self):
		for s in iter(self.dataloader):
			x = s  # to make sure s is actually used


#######################################
### CLASSES FOR TENSOWFLOW CONSUMER ###
#######################################


def dataloader_tensorflow_dataAPI(
	source: 'DataSourceBase', num_threads: int, batch_size: int, output_signature, pad_types
) -> tf.data.Dataset:
	"""
	Create tf.data.Dataset for TensorflowConsumer

	Args:
		source (DataSourceBase): data source with data for tf.data.Dataset
		num_threads (int): amount of threads to be used for loading
		batch_size (int): batch size to be used for loading
		output_signature: output signature of elements returned by source
		pad_types: type of elements returned by source to enable appropriate padding

	Returns:
		tf.data.Dataset: data set enabling parallel loading
	"""

	# create data set with relevant indices
	dataset = tf.data.Dataset.from_generator(
		lambda: list(range(len(source))), output_signature=tf.TensorSpec(None, tf.int32)
	)

	# shuffle data set
	dataset = dataset.shuffle(len(source))

	# enable parallel loading using dataset.map
	dataset = dataset.map(
		lambda i: tf.py_function(func=lambda x: source[x.numpy()], inp=[i], Tout=output_signature),
		num_parallel_calls=num_threads,
	)

	# batch data, with padding if necessary
	if pad_types is not None:
		dataset = dataset.padded_batch(batch_size, padded_shapes=pad_types, drop_remainder=True)
	else:
		dataset = dataset.batch(batch_size, drop_remainder=True)

	dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

	return dataset


class TensorflowConsumer:
	"""
	Class consuming data set generated by dataloader_tensorflow_API

	Attributes
	----------
	dataset : DataSourceBase
	    data set containing data to be loaded
	num_threads : int
		amount of threads which should be used for loading
	batch_size : int
		batch size with for loading
	device : str
		determines if data should be loaded to GPU or kept on CPU

	Methods
	-------
	__init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, prng: np.random.RandomState | None = None, shuffle: bool = False, **kwargs):
		creates TensorflowConsumer object
	__call__(self):
		iterate through entire data set using created tensorflow data set
	"""

	def __init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, device: str):
		# determine correct device
		gpu_devices = tf.config.list_logical_devices('GPU')
		cpu_devices = tf.config.list_logical_devices('CPU')
		if device == 'gpu' and len(gpu_devices) > 0:
			self.device = gpu_devices[0].name
		else:
			self.device = cpu_devices[0].name

		self.num_samples = (
			len(dataset) // batch_size
		)  # divide through batch size without remainder, since drop_remainder = true

		# create iterator out of data set
		with tf.device(self.device):
			self.iterator = iter(
				# create tensorflow data set
				dataloader_tensorflow_dataAPI(
					dataset,
					num_threads,
					batch_size,
					dataset.get_tf_output_signature(),
					dataset.get_tf_pad_types(),
				)
			)

	def __call__(self):
		# iterate through data set and load to appropriate device
		with tf.device(self.device):
			for _ in range(self.num_samples):
				s = self.iterator.get_next()


################################################
### CLASSES FOR TENSORFLOW SEQUENCE CONSUMER ###
################################################


class DataLoaderTensorflowSequence(tf.keras.utils.Sequence):
	"""
	Data loader based on tf.keras.utils.Sequence

	Attributes
	----------
	dataset : DataSourceBase
	    data set containing data to be loaded
	batch_size : int
		batch size with for loading
	prng : np.random.RandomState | None = None
		used to shuffle data if shuffle=True
	shuffle : bool = False
		if True data will be shuffled before iterating

	Methods
	-------
	__init__(self, dataset: 'DataSourceBase', batch_size: int, prng: np.random.RandomState | None = None, shuffle: bool = False, **kwargs):
		creates DataLoaderTensorflowSequence object
	__len__(self):
	    returns amount of batches in data set
	__getitem__(self, index: int):
		return batch index of data set
	"""

	def __init__(
		self,
		dataset: 'DataSourceBase',
		batch_size: int,
		prng: np.random.RandomState | None = None,
		shuffle: bool = False,
		**kwargs,
	):
		self.dataset = dataset
		self.batch_size = batch_size
		self.inds = np.arange(len(self.dataset))

		# shuffle data if necessary
		if prng is None:
			prng = np.random.RandomState(193)
		if shuffle:
			prng.shuffle(self.inds)

	def __len__(self):
		return int(len(self.dataset) // self.batch_size)

	def __getitem__(self, index: int):
		# convert elements of batch to tensors
		try:
			batch = [
				(
					self.dataset[self.inds[i]]
					if isinstance(self.dataset[self.inds[i]], tf.Tensor)
					else tf.convert_to_tensor(self.dataset[self.inds[i]])
				)
				for i in np.arange(index * self.batch_size, (index + 1) * self.batch_size)
			]
		# elements in batch have subelements, which need to be converted
		except Exception:
			batch = [
				[
					(x if isinstance(x, tf.Tensor) else tf.convert_to_tensor(x))
					for x in self.dataset[self.inds[i]]
				]
				for i in np.arange(index * self.batch_size, (index + 1) * self.batch_size)
			]
		return batch


class TensorflowSequenceConsumer:
	"""
	Class consuming data set using DataLoaderTensorflowSequence and OrderedEnqueur

	Attributes
	----------
	dataset : DataSourceBase
	    data set containing data to be loaded
	num_threads : int
		amount of threads which should be used for loading
	batch_size : int
		batch size with for loading
	device : str
		determines if data should be loaded to GPU or kept on CPU

	Methods
	-------
	__init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, prng: np.random.RandomState | None = None, shuffle: bool = False, **kwargs):
		creates TensorflowSequenceConsumer object
	__call__(self):
		iterate through entire data set using DataLoaderTensorflowSequence
	"""

	def __init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, device: str):
		# determine correct device
		gpu_devices = tf.config.list_logical_devices('GPU')
		cpu_devices = tf.config.list_logical_devices('CPU')
		if device == 'gpu' and len(gpu_devices) > 0:
			self.device = gpu_devices[0].name
		else:
			self.device = cpu_devices[0].name
		self.num_threads = num_threads

		# create dala loader and enqueuer on correct device
		with tf.device(self.device):
			dataloader = DataLoaderTensorflowSequence(dataset, batch_size=batch_size, shuffle=True)
			self.num_samples = len(dataloader)
			self.enqueuer = all_utils.OrderedEnqueuer(
				dataloader, use_multiprocessing=False, shuffle=True
			)

	def __call__(self):
		# iterate through data using OrderedEnqueur and load samples to correct device
		with tf.device(self.device):
			self.enqueuer.start(workers=self.num_threads)
			dataloader = self.enqueuer.get()
			for i, s in enumerate(dataloader):
				x = s  # to make sure s is actually used
				if i == self.num_samples:
					break
			self.enqueuer.stop()


########################################
##### CLASSES FOR PYTORCH CONSUMER #####
########################################


class DataSetPytorch(torch.utils.data.Dataset):
	"""
	Data set based on torch.utils.data.Dataset

	Attributes
	----------
	dataset : DataSourceBase
	    data set containing data to be loaded
	kwargs :
		further arguments passed to DataLoader

	Methods
	-------
	__init__(self, dataset: 'DataSourceBase', **kwargs):
		creates DataSetPytorch object
	__len__(self):
	    returns length of data set
	__iter__(self):
		iterates through data set
	__getitem__(self, index: int):
		return element index of data set
	"""

	def __init__(self, dataset: 'DataSourceBase', **kwargs):
		self.dataset = dataset
		self.dataloader = DataLoader(dataset, **kwargs)

	def __len__(self):
		return len(self.dataset)

	def __iter__(self):
		return self.dataloader.__iter__()

	def __getitem__(self, index: int):
		return self.dataset.__getitem__(index)


class PytorchConsumer:
	"""
	Class consuming data set using DataSetPytorch

	Attributes
	----------
	dataset : DataSourceBase
	    data set containing data to be loaded
	num_threads : int
		amount of threads which should be used for loading
	batch_size : int
		batch size with for loading
	device : str
		determines if data should be loaded to GPU or kept on CPU

	Methods
	-------
	__init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, prng: np.random.RandomState | None = None, shuffle: bool = False, **kwargs):
		creates PytorchConsumer object
	__call__(self):
		iterate through entire data set using DataSetPytorch
	"""

	def __init__(self, dataset: 'DataSourceBase', num_threads: int, batch_size: int, device: str):
		# determine correct device
		self.device = torch.device(
			'cuda' if (torch.cuda.is_available() and device == 'gpu') else 'cpu'
		)

		# create pytorch data set
		torch_dataset = DataSetPytorch(dataset)
		# create pytorch data loader using dataset
		self.dataloader = torch.utils.data.DataLoader(
			torch_dataset,
			num_workers=num_threads,
			batch_size=batch_size,
			collate_fn=dataset.get_collate_fn(),
			drop_last=True,
			shuffle=True,
		)
		self.num_samples = len(self.dataloader)

	def __call__(self):
		# iterate through samples and load tensors to correct device
		for s in iter(self.dataloader):
			for t in s:
				t.to(self.device)


#########################################
### CLASSES FOR DALI PYTORCH CONSUMER ###
#########################################


class DALIPytorchConsumer:
	"""
	Class consuming data set using NVIDIA DALI PyTorch plugin

	Attributes
	----------
	pipeline : Callable
		DALI pipeline to be used for loading
	pipe_args : tuple
		tuple of arguments that need to be passed to pipeline
	labels : list
		labels of features of returned samples
	num_threads : int
		amount of threads to be used for loading
	batch_size : int
		batch size to be used for loading
	device : str
		device to which data should be loaded

	Methods
	-------
	__init__(self, pipeline: Callable, pipe_args: tuple, labels: list, num_threads: int, batch_size: int, device: str):
		creates DALIPytorchConsumer object
	__call__(self):
		iterate through entire data set using DALIs PyTorch plugin
	"""

	def __init__(
		self,
		pipeline: Callable,
		pipe_args: tuple,
		labels: list,
		num_threads: int,
		batch_size: int,
		device: str,
	):
		# set according device id
		self.device_id = None
		if device == 'gpu':
			self.device_id = torch.cuda.current_device()

		# initialize pipeline
		self.pipeline = pipeline(
			*pipe_args,
			device,
			batch_size=batch_size,
			num_threads=num_threads,
			device_id=self.device_id,
		)

		# create iterator from pipeline
		self.dataloader = DALIGenericIterator(
			[self.pipeline],
			labels,
			prepare_first_batch=False,
			reader_name='reader' + labels[0],
			last_batch_policy=LastBatchPolicy.DROP,
		)

	def __call__(self):
		for s in iter(self.dataloader):
			x = s  # make sure s is actually used


############################################
### CLASSES FOR DALI TENSORFLOW CONSUMER ###
############################################


class DALITensorflowConsumer:
	"""
	Class consuming data set using NVIDIA DALI TensorFlow plugin

	Attributes
	----------
	pipeline : Callable
		DALI pipeline to be used for loading
	pipe_args : tuple
		tuple of arguments that need to be passed to pipeline
	num_threads : int
		amount of threads to be used for loading
	batch_size : int
		batch size to be used for loading
	device : str
		device to which data should be loaded
	shapes : list
		list with shapes of elements returned by pipeline
	dtypes : list
		list with dtypes of elements returned by pipeline

	Methods
	-------
	__init__(self, pipeline: Callable, pipe_args: tuple, num_threads: int, batch_size: int, device: str, shapes: list, dtypes: list):
		creates DALIPytorchConsumer object
	__call__(self):
		iterate through entire data set using DALIs TensorFlow plugin
	"""

	def __init__(
		self,
		pipeline: Callable,
		pipe_args: tuple,
		num_threads: int,
		batch_size: int,
		device: str,
		shapes: list,
		dtypes: list,
	):
		# set according device and device id
		gpu_devices = tf.config.list_logical_devices('GPU')
		cpu_devices = tf.config.list_logical_devices('CPU')
		self.device_id = None
		if device == 'gpu':
			self.device_id = torch.cuda.current_device()
			self.tf_device = gpu_devices[0].name
		else:
			self.tf_device = cpu_devices[0].name

		self.num_threads = num_threads
		self.batch_size = batch_size
		self.shapes = shapes
		self.dtypes = dtypes

		# initialize pipeline
		self.pipeline = pipeline(
			*pipe_args,
			device,
			batch_size=self.batch_size,
			num_threads=self.num_threads,
			device_id=self.device_id,
		)
		self.pipeline.build()
		self.len = list(self.pipeline.reader_meta().values())[0]['epoch_size'] // self.batch_size
		# create dali tf iterator
		self.daliop = dali_tf.DALIIterator()

	def __call__(self):
		# iterate through data set and make sure tensors end up on according device
		with tf.device(self.tf_device):
			for _ in range(self.len):
				s = self.daliop(
					pipeline=self.pipeline,
					batch_size=self.batch_size,
					num_threads=self.num_threads,
					device_id=self.device_id,
					shapes=self.shapes,
					dtypes=self.dtypes,
				)
