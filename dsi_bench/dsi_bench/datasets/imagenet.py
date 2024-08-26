import h5py
import zarr
import pickle
import numpy as np
import cv2
import torch
import torchvision
import tensorflow as tf
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import os
from pathlib import Path
from tqdm import tqdm
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
		the base path of the dataset (must contain images directly)
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	DIMENSION = (224, 224, 3)
		dimension of returned images

	Methods
	-------
	__init__(self, path: Path | str, amount: int = -1):
		creates DataSourceBase object
	check_data(self):
		not implemented for DataSourceBase
	__len__(self):
		not implemented for DataSourceBase
	__getitem__(self, index: int):
		not implemented for DataSourceBase
	_resize(self, img: np.ndarray):
		resize images to DIMENSION when calling __getitem__ to enable batching
	get_collate_fn(self):
		collate function not required for resized ImageNet so returns None
	get_tf_output_signature(self):
		returns a tuple of tf.TensorSpec specifying format of samples
	get_tf_pad_types(self):
		padding not required for resized ImageNet so returns None
	set_return_type(self, return_type: str):
		sets return type to return_type for __getitem__ of child classes
	get_available_return_types(self):
		returns list of implemented return_types
	"""

	DIMENSION = (224, 224, 3)

	def __init__(self, path: Path | str, amount: int = -1):
		self.path = Path(path)
		self.amount = amount
		self.return_type = 'python'
		self.check_data()

	def check_data(self):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError

	def __getitem__(self, index: int):
		raise NotImplementedError

	def _resize(self, img: np.ndarray):
		resized_img = cv2.resize(img, dsize=self.DIMENSION[:2])

		# only one channel, expand to 3
		if len(resized_img.shape) == 2:
			resized_img = np.expand_dims(resized_img, axis=-1)

		if len(resized_img.shape) == 3 and resized_img.shape[2] == 1:
			resized_img = np.repeat(resized_img, 3, axis=-1)

		return resized_img.astype(np.uint8)

	def get_collate_fn(self):
		return None

	def get_tf_output_signature(self):
		# pixel values are uint8
		return tf.TensorSpec(DataSourceBase.DIMENSION, tf.uint8)

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
	In this case, we are loading from a individual JPEG files

	Attributes
	----------
	path : Path | str
		the base path of the dataset (must contain images directly)
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	Methods
	-------
	__init__(self, path: Path | str, filename: str, key: list | str):
		creates DataSource object
	check_data(self):
		get list of JPEG files and count to get len
	__len__(self):
		returns amount of samples and calls check data if not present
	__getitem__(self, index: int):
		returns resized image <index>
		always acts as if return_type == python regardless of actual value
	"""

	def __init__(self, path: Path | str, amount: int = -1):
		super().__init__(path, amount)

	def prepare(self):
		pass

	def check_data(self):
		try:
			file_list = sorted(
				list(filter(lambda f: f.endswith('.JPEG'), os.listdir(self.path)))
			)  # sort for determinism

			# parse list of files and get len
			if self.amount > 0:
				self.len = self.amount
				self.file_list = file_list[0 : self.amount]
			else:
				self.len = len(file_list)
				self.file_list = file_list
		except Exception:
			raise DatasetNotAvailableError('Failed to load dataset')

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# get correct filename and read image
		path = self.path / self.file_list[idx]
		img = cv2.imread(path.absolute().as_posix(), cv2.IMREAD_UNCHANGED)

		# return resized image
		return self._resize(img)


#################################
####### RAW STORAGE FORMAT ######
#################################


class DataSourceRaw(DataSource):
	"""
	This class contains functions to load the data using it's original (raw) storage format.
	In this case, we are loading from a individual JPEG files

	Attributes
	----------
	path : Path | str
		the base path of the dataset (must contain images directly)
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	Methods
	-------
	__getitem__(self, index: int):
		returns resized image <index>
		adapts content of tuple depending on set return type
	"""

	def __getitem__(self, index: int):
		idx = index % self.len
		path = self.path / self.file_list[idx]

		# convert sample to according return type
		if self.return_type == 'pytorch':
			img = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)
			return torch.permute(img, (1, 2, 0))  # make sure it is RGB not BRG
		elif self.return_type == 'tensorflow':
			img = tf.io.read_file(path.absolute().as_posix())
			img = tf.image.decode_jpeg(img, channels=self.DIMENSION[2])
			img.set_shape(self.DIMENSION)
			return img
		else:
			return cv2.imread(path.absolute().as_posix())


def convert_to_raw(source: DataSource, output_path: Path | str) -> DataSourceRaw:
	"""
	convert DataSource to format raw and return according DataSourceRaw object

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceRaw: reads data in raw format from output_path
	"""

	output_dir = Path(output_path)

	try:
		return DataSourceRaw(output_dir, source.amount)
	except DatasetNotAvailableError:
		print('copying raw dataset')

		os.makedirs(output_dir, exist_ok=True)

		# save individual JPEGs in new location
		for idx in tqdm(range(len(source)), total=len(source)):
			img = source[idx]
			file_path = output_dir / source.file_list[idx]
			cv2.imwrite(file_path, img)

		return DataSourceRaw(output_dir, source.amount)


#################################
###### HDF5 STORAGE FORMAT ######
#################################


class DataSourceHDF5(DataSourceBase):
	"""
	This class contains functions to load the data using the hdf5 storage format.

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	H5_FILE_NAME = 'imagenet.h5'
		name of hdf5 file relative to `path` containing image data

	Methods
	-------
	check_data(self):
		sets len based on samples in HDF5 file
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns resized image <index>
		adapts content of tuple depending on set return type
	"""

	H5_FILE_NAME = 'imagenet.h5'

	def __init__(self, path: Path | str, amount: int = -1):
		super().__init__(path, amount)

	def check_data(self):
		try:
			with h5py.File(self.path / self.H5_FILE_NAME, 'r') as fh:
				self.len = len(fh.keys())
		except Exception:
			raise DatasetNotAvailableError('Failed to load dataset')

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = str(index % self.len)

		# read data of image <index>
		with h5py.File(self.path / self.H5_FILE_NAME, 'r') as fh:
			img = fh[idx][0, ...]

		# convert sample to according return type
		if self.return_type == 'pytorch':
			return torch.from_numpy(img)
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(img, dtype=tf.uint8)
		else:
			return img


def convert_to_hdf5(source: DataSource, output_path: Path | str) -> DataSourceHDF5:
	"""
	convert DataSource to format hdf5 and return according DataSourceHDF5 object

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceHDF5: reads data in hdf5 format from output_path
	"""

	output_dir = Path(output_path)
	output_file = output_dir / DataSourceHDF5.H5_FILE_NAME

	try:
		return DataSourceHDF5(output_dir, source.amount)
	except DatasetNotAvailableError:
		print('converting to hdf5')

		os.makedirs(output_dir, exist_ok=True)

		# save image data as hdf dataset
		with h5py.File(output_file, 'w') as h5_file:
			for idx in tqdm(range(len(source)), total=len(source)):
				img = source[idx]
				dataset = h5_file.create_dataset(str(idx), shape=(1, *img.shape), dtype=np.uint8)
				dataset[0, ...] = img
				h5_file.flush()
			h5_file.close()

		return DataSourceHDF5(output_dir, source.amount)


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
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	ZARR_FILE_NAME = 'imagenet.zarr'
		name of zarr file relative to `path` containing image data

	Methods
	-------
	check_data(self):
		sets len and checks if all required keys are present in zarr file
	__len__(self):
		returns length of data set
	__getitem__(self, index: int):
		returns resized image <index>
		adapts content of tuple depending on set return type
	"""

	ZARR_FILE_NAME = 'imagenet.zarr'

	def __init__(self, path: Path | str, amount: int = -1):
		super().__init__(path, amount)

	def check_data(self):
		try:
			data = zarr.open_group(self.path / self.ZARR_FILE_NAME, mode='r')
			self.len = len(data.keys())
		except zarr.errors.GroupNotFoundError:
			raise DatasetNotAvailableError
		return

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# read data of image <index>
		data = zarr.open_group(self.path / self.ZARR_FILE_NAME, mode='r')
		img = data[idx][...]

		# convert sample according to return type
		if self.return_type == 'pytorch':
			return torch.from_numpy(np.array(img))
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(img)
		else:
			return img


def convert_to_zarr(source: DataSource, output_path: Path | str) -> DataSourceZarr:
	"""
	convert DataSource to format zarr and return according DataSourceZarr object

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceZarr: reads data in zarr format from output_path
	"""

	output_dir = Path(output_path)

	try:
		return DataSourceZarr(output_dir, source.amount)
	except DatasetNotAvailableError:
		print('converting to zarr')

		os.makedirs(output_dir, exist_ok=True)

		# save image data in zarr group
		dest = zarr.open_group(output_dir / DataSourceZarr.ZARR_FILE_NAME, mode='w')
		for idx in tqdm(range(len(source)), total=len(source)):
			dest[idx] = source[idx]

		return DataSourceZarr(output_dir, source.amount)


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
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	PICKLE_DIR_NAME = 'imagenet.picklestore'
		directory relative to `path` containing individual pickle files

	Methods
	-------
	check_data(self):
		check if pickle directory exists and set len to amount of files found
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns resized image <index>
		adapts content of tuple depending on set return type
	"""

	PICKLE_DIR_NAME = 'imagenet.picklestore'

	def __init__(self, path: Path | str, amount: int = -1):
		super().__init__(path, amount)

	def check_data(self):
		if not (self.path / self.PICKLE_DIR_NAME).exists():
			raise DatasetNotAvailableError

		self.len = len(list((self.path / self.PICKLE_DIR_NAME).glob('*.pickle')))

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# read image data from corresponding pickle file
		with open(self.path / self.PICKLE_DIR_NAME / f'{idx}.pickle', 'rb') as f:
			img = pickle.load(f)

		# convert sample to according return type
		if self.return_type == 'pytorch':
			return torch.from_numpy(img)
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(img, dtype=tf.uint8)
		else:
			return img


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

	try:
		return DataSourcePickle(output_path, source.amount)
	except DatasetNotAvailableError:
		print('converting to pickle')

		os.makedirs(output_path / DataSourcePickle.PICKLE_DIR_NAME, exist_ok=True)

		# save individual images in individual pickle files
		for i in tqdm(range(len(source)), total=len(source)):
			with open(output_path / DataSourcePickle.PICKLE_DIR_NAME / f'{i}.pickle', 'wb') as f:
				# save using the highest protocol available
				pickle.dump(source[i], f, pickle.HIGHEST_PROTOCOL)

		return DataSourcePickle(output_path, source.amount)


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
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	NPY_DIR_NAME = 'imagenet.npy'
		directory relative to `path` containing npy files

	Methods
	-------
	check_data(self):
		sets len and checks if directory with npy files is present
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns resized image <index>
		adapts content of tuple depending on set return type
	"""

	NPY_DIR_NAME = 'imagenet.npy'

	def __init__(self, path: Path | str, amount: int = -1):
		super().__init__(path, amount)

	def check_data(self):
		if not (self.path / self.NPY_DIR_NAME).exists():
			raise DatasetNotAvailableError

		self.len = len(list((self.path / self.NPY_DIR_NAME).glob('*.npy')))

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# load image data from according npy file
		img = np.load(self.path / self.NPY_DIR_NAME / f'{idx}.npy')

		# convert sample according to return type
		if self.return_type == 'pytorch':
			return torch.from_numpy(img)
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(img, dtype=tf.uint8)
		else:
			return img


def convert_to_npy(source: DataSource, output_path: Path | str) -> DataSourceNPY:
	"""
	convert DataSource to format npy and return according DataSourceNPY object
	every image id saved in an individual npy file

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceNPY: reads data in npy format from output_path
	"""

	output_path = Path(output_path)

	try:
		return DataSourceNPY(output_path, source.amount)
	except DatasetNotAvailableError:
		print('converting to npy')

		os.makedirs(output_path / DataSourceNPY.NPY_DIR_NAME, exist_ok=True)

		# save images to individual npy files
		for i in tqdm(range(len(source)), total=len(source)):
			np.save(output_path / DataSourceNPY.NPY_DIR_NAME / f'{i}.npy', source[i])

		return DataSourceNPY(output_path, source.amount)


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
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	NPZ_DIR_NAME = 'imagenet.npz'
		directory relative to `path` containing npz files

	Methods
	-------
	check_data(self):
		sets len and checks if directory with np\ files is present
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns resized image <index>
		adapts content of tuple depending on set return type
	"""

	NPZ_DIR_NAME = 'imagenet.npz'

	def __init__(self, path: Path | str, amount: int = -1):
		super().__init__(path, amount)

	def check_data(self):
		if not (self.path / self.NPZ_DIR_NAME).exists():
			raise DatasetNotAvailableError

		self.len = len(list((self.path / self.NPZ_DIR_NAME).glob('*.npz')))

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# load image data from corresponding npz file
		data = np.load(self.path / self.NPZ_DIR_NAME / f'{idx}.npz')
		img = data['arr_0'].astype(np.uint8, copy=False)

		# convert sample to according return type
		if self.return_type == 'pytorch':
			return torch.from_numpy(img)
		elif self.return_type == 'tensorflow':
			return tf.convert_to_tensor(img, dtype=tf.uint8)
		else:
			return img


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

	try:
		return DataSourceNPZ(output_path, source.amount)
	except DatasetNotAvailableError:
		print('converting to compressed npz')

		os.makedirs(output_path / DataSourceNPZ.NPZ_DIR_NAME, exist_ok=True)

		# save images to individual npz files
		for i in tqdm(range(len(source)), total=len(source)):
			np.savez_compressed(output_path / DataSourceNPZ.NPZ_DIR_NAME / f'{i}.npz', source[i])

		return DataSourceNPZ(output_path, source.amount)


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
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	TFRECORD_FILE_NAME = 'imagenet.tfrecord'
		name of tfrecord file containing data. relative to `path`

	Methods
	-------
	check_data(self):
		checks if tfrecord file exists sets len based on amount of samples in tfrecord file
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns resized image <index>
		adapts content of tuple depending on set return type
	"""

	TFRECORD_FILE_NAME = 'imagenet.tfrecord'

	def __init__(self, path: Path | str, amount: int = -1):
		# set device to CPU to avoid weird errors in GPU environment during data loading
		self.device = tf.config.list_logical_devices('CPU')[0].name
		super().__init__(path, amount)

	def check_data(self):
		if not (self.path / self.TFRECORD_FILE_NAME).exists():
			raise DatasetNotAvailableError

		with tf.device(self.device):
			self.dataset = tf.data.TFRecordDataset(
				self.path / self.TFRECORD_FILE_NAME, num_parallel_reads=tf.data.AUTOTUNE
			)
			self.len = sum(1 for _ in self.dataset)  # calculate length of data set

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		with tf.device(self.device):
			# skip beginning of data set to retrieve image <index>
			elem = None
			for data in self.dataset.skip(idx).take(1):
				elem = data

			# create parsing dictionary to get serialized tensor of image
			parse_dic = {'img': tf.io.FixedLenFeature([], tf.string)}
			serialized_bytes = tf.io.parse_single_example(elem, parse_dic)
			feature = serialized_bytes['img']

			# decode image and convert to return type
			if self.return_type == 'pytorch':
				return torch.from_numpy(tf.io.decode_jpeg(feature, channels=3).numpy())
			elif self.return_type == 'tensorflow':
				return tf.io.decode_jpeg(feature, channels=3)
			else:
				return tf.io.decode_jpeg(feature, channels=3).numpy()


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

	try:
		return DataSourceTFRecordSingle(output_path, source.amount)
	except DatasetNotAvailableError:
		print('converting to tfrecord single')

		os.makedirs(output_path, exist_ok=True)
		fname_out = (
			(output_path / DataSourceTFRecordSingle.TFRECORD_FILE_NAME).absolute().as_posix()
		)

		# save image data into single tfrecord file
		with tf.io.TFRecordWriter(fname_out) as tf_writer:
			for idx in tqdm(range(len(source)), total=len(source)):
				img = source[idx]

				# make sure also BW images have 3 channels
				if len(source[idx].shape) == 2:
					img = np.expand_dims(img, axis=-1)

				# encode image as JPEG
				sample = {
					'img': tf.train.Feature(
						bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(img).numpy()])
					)
				}

				# save sample to file
				record_bytes = tf.train.Example(
					features=tf.train.Features(feature=sample)
				).SerializeToString()
				tf_writer.write(record_bytes)

		return DataSourceTFRecordSingle(output_path, source.amount)


######################################
### TFRECORD_SINGLE STORAGE FORMAT ###
######################################


class DataSourceTFRecordMulti(DataSourceBase):
	"""
	This class contains functions to load the data using the tfrecord_multi storage format

	Attributes
	----------
	path : Path | str
		the base path of the dataset
	amount : int = -1
		the amount of images which should be read from path. If -1 all will be used

	TFRECORD_DIR_NAME = 'imagenet.tfrecord'
		directory relative to `path` containing tfrecord files

	Methods
	-------
	check_data(self):
		checks if directory of files exists and sets len based on amount of tfrecord files
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns resized image <index>
		adapts content of tuple depending on set return type
	"""

	TFRECORD_DIR_NAME = 'imagenet.tfrecord'

	def __init__(self, path: Path | str, amount: int = -1):
		# set device to CPU to avoid weird errors in GPU environment during data loading
		self.device = tf.config.list_logical_devices('CPU')[0].name
		super().__init__(path, amount)

	def check_data(self):
		if not (self.path / self.TFRECORD_DIR_NAME).exists():
			raise DatasetNotAvailableError

		self.len = len(
			list((self.path / self.TFRECORD_DIR_NAME).glob('*.tfrecord'))
		)  # count number of images

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		with tf.device(self.device):
			# read sample
			dataset = tf.data.TFRecordDataset(
				self.path / self.TFRECORD_DIR_NAME / f'{idx}.tfrecord',
				num_parallel_reads=tf.data.AUTOTUNE,
			)
			elem = None
			for data in dataset.take(1):
				elem = data

			# create parsing dictionary to get serialized tensor of image
			parse_dic = {'img': tf.io.FixedLenFeature([], tf.string)}
			serialized_bytes = tf.io.parse_single_example(elem, parse_dic)
			feature = serialized_bytes['img']

			# decode image and convert to return type
			if self.return_type == 'pytorch':
				return torch.from_numpy(tf.io.decode_jpeg(feature, channels=3).numpy())
			elif self.return_type == 'tensorflow':
				return tf.io.decode_jpeg(feature, channels=3)
			else:
				return tf.io.decode_jpeg(feature, channels=3).numpy()


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

	try:
		return DataSourceTFRecordMulti(output_path, source.amount)
	except DatasetNotAvailableError:
		print('converting to tfrecord multi')

		os.makedirs(output_path / DataSourceTFRecordMulti.TFRECORD_DIR_NAME, exist_ok=True)

		# convert individual samples to tfrecord files
		for idx in tqdm(range(len(source)), total=len(source)):
			fname_out = (
				(output_path / DataSourceTFRecordMulti.TFRECORD_DIR_NAME / f'{idx}.tfrecord')
				.absolute()
				.as_posix()
			)
			with tf.io.TFRecordWriter(fname_out) as tf_writer:
				img = source[idx]

				# make sure also BW images have 3 channels
				if len(source[idx].shape) == 2:
					img = np.expand_dims(img, axis=-1)

				# encode image as JPEG
				sample = {
					'img': tf.train.Feature(
						bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(img).numpy()])
					)
				}

				# save sample to own file
				record_bytes = tf.train.Example(
					features=tf.train.Features(feature=sample)
				).SerializeToString()
				tf_writer.write(record_bytes)

		return DataSourceTFRecordMulti(output_path, source.amount)


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
###### RAW DALI PIPELINE ######
###############################


def dali_raw_prepare(source: DataSource, output_path: str | Path) -> tuple:
	"""
	make sure all conversions are done and return arguments required for raw pipeline

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		tuple: tuple containing arguments for raw pipeline
	"""

	# make sure data is converted
	raw = get_available_converters()['raw'](source, output_path)

	# return path to images and list of files
	return (raw.path, os.listdir(raw.path))


@pipeline_def
def dali_raw_pipeline(file_path: str | Path, file_list: list, device: str):
	"""
	DALI pipeline definition using file reader

	Args:
		file_path (str | Path): root directory of jpeg files
		file_list (list): list of jpeg files
		device (str): device to load samples to
	"""

	# read content of jpeg file
	inputs, labels = fn.readers.file(
		file_root=file_path,
		files=file_list,
		random_shuffle=True,
		seed=193,
		lazy_init=True,
		name='readerimg',
	)

	# image decoder only supports mixed not gpu
	device = 'mixed' if device == 'gpu' else 'cpu'

	# decode and return image data
	return fn.decoders.image(inputs, device=device, output_type=types.RGB)


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
	file_path = npy.path / npy.NPY_DIR_NAME

	# list of all relevant npy files
	file_list = [str(i) + '.npy' for i in range(len(npy))]

	return (file_path, file_list)


@pipeline_def
def dali_npy_pipeline(file_path: str | Path, file_list: list, device: str):
	"""
	DALI pipeline definition using numpy reader

	Args:
		file_path (str | Path): root directory of npy files
		file_list (list): list of numpy files in subfolders
		device (str): device to load samples to
	"""

	# read image data
	sample = fn.readers.numpy(
		device=device,
		file_root=file_path,
		files=file_list,
		name='readerimg',
		random_shuffle=True,
		seed=193,
		lazy_init=True,
	)

	return sample


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

	# path to tfrecord file
	file_path = tf_single.path / tf_single.TFRECORD_FILE_NAME

	# create index file required by tfrecord reader
	idx_path = tf_single.path / 'idx_files'
	if not idx_path.is_dir():
		idx_path.mkdir(parents=True, exist_ok=True)

	idx_path = idx_path / 'imagenet.idx'
	if not idx_path.is_file():
		call(['tfrecord2idx', file_path, idx_path])

	return (file_path, idx_path)


@pipeline_def
def dali_tfrecord_single_pipeline(file_path: str | Path, idx_path: str | Path, device: str):
	"""
	DALI pipeline definition using tfrecord reader

	Args:
		file_path (str | Path): path to tfrecord file
		idx_path (str | Path): path to index file
		keys (list): relevant keys of features
		device (str): device to load samples to
	"""

	# define what type of features we have
	# string since image is encoded are serialized
	features = {'img': tfrec.FixedLenFeature([], tfrec.string, '')}

	# read image from file
	inputs = fn.readers.tfrecord(
		path=file_path,
		index_path=idx_path,
		features=features,
		random_shuffle=True,
		seed=193,
		lazy_init=True,
		name='readerimg',
	)

	# image decoder only supports mixed not gpu
	device = 'mixed' if device == 'gpu' else 'cpu'

	return fn.decoders.image(inputs['img'], device=device, output_type=types.RGB)


##################################
#### AVAILABLE DALI PIPELINES ####
##################################


def get_available_pipelines():
	pipelines = {
		'raw': {
			'preparation': dali_raw_prepare,
			'pipeline': dali_raw_pipeline,
			'labels': ['img'],
			'pytorch_args': {},
			'tensorflow_args': {'shapes': [()], 'dtypes': [tf.uint8]},
		},
		'npy': {
			'preparation': dali_npy_prepare,
			'pipeline': dali_npy_pipeline,
			'labels': ['img'],
			'pytorch_args': {},
			'tensorflow_args': {'shapes': [()], 'dtypes': [tf.uint8]},
		},
		'tfrecord_single': {
			'preparation': dali_tfrecord_single_prepare,
			'pipeline': dali_tfrecord_single_pipeline,
			'labels': ['img'],
			'pytorch_args': {},
			'tensorflow_args': {'shapes': [()], 'dtypes': [tf.uint8]},
		},
	}

	return pipelines
