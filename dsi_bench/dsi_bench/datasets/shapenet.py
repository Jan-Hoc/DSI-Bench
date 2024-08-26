import h5py
import zarr
import pickle
import numpy as np
import torch
import pytorch3d
import pytorch3d.io
import pytorch3d.datasets
import trimesh
import tensorflow as tf
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import os
from pathlib import Path
from tqdm import tqdm
from subprocess import call
import json
import shutil
import random


class DatasetNotAvailableError(Exception):
	pass


class ReturnTypeNotSupportedError(Exception):
	pass


def parse_json(json_path) -> dict:
	"""
	function to parse OBJ JSON

	Args:
		json_path: path of obj file

	Returns:
		dict: content of JSON
	"""
	with open(json_path, 'r') as f:
		data = json.load(f)
	return data


class ShapeNetCoreParse:
	"""
	Class to read original ShapeNet data set
	"""

	def __init__(self, path_shapenetcore: Path | str):
		"""path_shapenetcore = '/mydata/fastpointstomesh/firat/projects/ShapeNetCore'"""
		self.path_shapenetcore = path_shapenetcore
		shapenet_id_name = os.path.join(
			path_shapenetcore, 'shapenet_synset_list'
		)  # downloaded from https://gist.githubusercontent.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e/raw/5064af3603d509b79229f6931998d4e197575ad3/shapenet_synset_list
		self.id2name = self._parse_shapenet_idnames(shapenet_id_name, verbose=0)
		self.synset_name2id = {v: k for k, v in self.id2name.items()}
		self.id_subdirs = self._iterate_files(verbose=0)
		self.synset_num_models = self._count_synset_models()
		self.id_meta = self._get_meta_json_obj_all(self.id_subdirs, verbose=0)
		self.id_obj = self._get_obj_all(self.id_subdirs, verbose=0)

	def _parse_shapenet_idnames(self, shapenet_id_name: Path | str, verbose: int = 0):
		"""Parses the shapenet synset list file and returns a dictionary of id to name mapping."""
		out_shapenet_id_name_npy = os.path.join(self.path_shapenetcore, 'shapenet_synset_list.npy')
		if os.path.isfile(out_shapenet_id_name_npy):
			if verbose:
				print(f'Loading id2name from file: {out_shapenet_id_name_npy}')
			id2name = np.load(out_shapenet_id_name_npy, allow_pickle=True).item()
			return id2name
		id2name = {}
		with open(shapenet_id_name, 'r') as fh:
			while True:
				s = fh.readline()
				if len(s) == 0:
					break
				s_l = str(s).split(' ', 1)
				id, name = s_l[0], ' '.join(s_l[1:])
				name = name.replace(' \n', '')
				name = name.replace('\n', '')
				id2name[id] = name
		np.save(out_shapenet_id_name_npy, id2name)
		if verbose:
			print(f'Saved id2name for quick read in future to file: {out_shapenet_id_name_npy}')
		return id2name

	def _count_synset_models(self):
		"""returns number of models existing under each synset."""
		synset_num_models = {}
		for k, v in self.synset_name2id.items():
			l_models = self.id_subdirs[v]
			if isinstance(l_models, float) and np.isnan(l_models):
				synset_num_models[k] = 0
			else:
				synset_num_models[k] = len(l_models)
		return synset_num_models

	def _get_obj(self, dir_unique_id: Path | str):
		"""Expects dir_unique_id=os.path.join(path_shapenetcore, id, id_subdirs[id][it]) as input. Returns path to obj, not mesh data"""
		obj_obj = os.path.join(dir_unique_id, 'models', 'model_normalized.obj')
		return obj_obj

	def _get_obj_all(self, id_subdirs: dict, verbose: int = 0):
		"""Expects id_subdirs as input. Returns a dictionary of id to obj mapping."""
		tmp_filename = os.path.join(self.path_shapenetcore, 'id_obj.npy')
		if os.path.isfile(tmp_filename):
			if verbose:
				print(f'Loading id_obj from file: {tmp_filename}')
			id_obj = np.load(tmp_filename, allow_pickle=True).item()
			return id_obj
		id_obj = {}
		if verbose:
			pbar = tqdm(
				total=len(id_subdirs.keys()),
				desc='Parsing ShapeNetCore OBJ files',
				position=0,
				leave=True,
			)
		for id in id_subdirs.keys():
			if isinstance(id_subdirs[id], float) and np.isnan(id_subdirs[id]):
				id_obj[id] = np.nan
				if verbose:
					pbar.update(1)
				continue
			id_obj[id] = {}
			for i in range(len(id_subdirs[id])):
				it = id_subdirs[id][i]
				dir_unique_id = os.path.join(self.path_shapenetcore, id, it)
				obj_obj = self._get_obj(dir_unique_id)
				id_obj[id][it] = obj_obj
			if verbose:
				pbar.update(1)
		if verbose:
			pbar.close()
		np.save(tmp_filename, id_obj)
		if verbose:
			print(f'Saved id_obj for quick read in future to file: {tmp_filename}')
		return id_obj

	def _get_meta_json_obj(self, dir_unique_id: Path | str):
		"""Expects dir_unique_id=os.path.join(path_shapenetcore, id, id_subdirs[id][it]) as input. Returns actual JSON data, not path"""
		obj_json = os.path.join(dir_unique_id, 'models', 'model_normalized.json')
		obj_meta = parse_json(obj_json)
		return obj_meta

	def _get_meta_json_obj_all(self, id_subdirs: list, verbose: int = 0):
		tmp_filename = os.path.join(self.path_shapenetcore, 'id_meta.npy')
		if os.path.isfile(tmp_filename):
			if verbose:
				print(f'Loading id_meta from file: {tmp_filename}')
			id_meta = np.load(tmp_filename, allow_pickle=True).item()
			return id_meta
		id_meta = {}
		if verbose:
			pbar = tqdm(
				total=len(id_subdirs.keys()),
				desc='Parsing ShapeNetCore JSON files',
				position=0,
				leave=True,
			)
		for id in id_subdirs.keys():
			if isinstance(id_subdirs[id], float) and np.isnan(id_subdirs[id]):
				id_meta[id] = np.nan
				if verbose:
					pbar.update(1)
				continue
			id_meta[id] = {}
			for i in range(len(id_subdirs[id])):
				it = id_subdirs[id][i]
				dir_unique_id = os.path.join(self.path_shapenetcore, id, it)
				obj_meta = self._get_meta_json_obj(dir_unique_id)
				id_meta[id][it] = obj_meta
			if verbose:
				pbar.update(1)
		if verbose:
			pbar.close()
		np.save(tmp_filename, id_meta)
		if verbose:
			print(f'Saved id_meta for quick read in future to file: {tmp_filename}')
		return id_meta

	def _has_obj_in_dir(self, dirname: Path | str):
		"""Checks if the directory has an obj file in expected location."""
		fname = os.path.join(dirname, 'models', 'model_normalized.obj')
		if os.path.isfile(fname):
			return True
		return False

	def _iterate_files(self, verbose: int = 0):
		tmp_filename = os.path.join(self.path_shapenetcore, 'id_subdirs.npy')
		if os.path.isfile(tmp_filename):
			if verbose:
				print(f'Loading id_subdirs from file: {tmp_filename}')
			id_subdirs = np.load(tmp_filename, allow_pickle=True).item()
			return id_subdirs
		id_subdirs = {}
		if verbose:
			pbar = tqdm(
				total=len(self.id2name.keys()),
				desc='Iterating over ShapeNetCore categories',
				position=0,
				leave=True,
			)
		for id in self.id2name.keys():
			id_subdirs[id] = np.nan
			if os.path.isdir(os.path.join(self.path_shapenetcore, id)):
				fl = os.listdir(os.path.join(self.path_shapenetcore, id))
				for item in fl:
					if not self._has_obj_in_dir(os.path.join(self.path_shapenetcore, id, item)):
						fl.remove(item)
				if len(fl) > 0:
					id_subdirs[id] = fl
			if verbose:
				pbar.update(1)
		if verbose:
			pbar.close()
		np.save(tmp_filename, id_subdirs)
		if verbose:
			print(f'Saved id_subdirs for quick read in future to file: {tmp_filename}')
		return id_subdirs

	def _demo_get_item(self):
		"""Demo function to show how to get an obj from the ShapeNetCore dataset."""
		synset_id = self.synset_name2id['bottle']
		model_id = self.id_subdirs[synset_id][0]
		obj_fname = self.id_obj[synset_id][model_id]
		obj = None  # load mesh from obj file (trimesh.load_mesh, pytorch3d.io.load_objs_as_meshes, tfg.io.triangle_mesh.load, etc.)
		sample = tuple(obj, synset_id)


class DataSourceBase:
	"""
	Base class for all data sources. This class is not meant to be used directly.

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	DATA_KEYS = {
		'synset_id': tf.string,
		'model_id': tf.string,
		'verts': tf.float32,
		'faces': tf.float32,
		'label': tf.string,
	}

	Methods
	-------
	__init__(self, path: Path | str, filename: str, key: list | str):
		creates DataSourceBase object
	_create_index_mapping(self):
		creates mapping from index to (synset_id, model_id) pairs
	check_data(self):
		not implemented for DataSourceBase
	__len__(self):
		not implemented for DataSourceBase
	__getitem__(self, index: int):
		not implemented for DataSourceBase
	get_collate_fn(self):
		returns collate function to enable batching with pytorch
	get_tf_output_signature(self):
		returns a tuple of tf.TensorSpec specifying format of samples
	get_tf_pad_types(self):
		returns padding type to enable batching with tensorflow
	set_return_type(self, return_type: str):
		sets return type to return_type for __getitem__ of child classes
	get_available_return_types(self):
		returns list of implemented return_types
	"""

	# keys of pytorch3d.datasets.ShapeNet which are returned
	DATA_KEYS = {
		'synset_id': tf.string,
		'model_id': tf.string,
		'verts': tf.float32,
		'faces': tf.float32,
		'label': tf.string,
	}

	def __init__(self, path: Path | str, key: list | str | None = None):
		# path to original data set to allow creation of various helper functions (e.g. synset name to synset id mapping)
		parse_path = path
		if hasattr(self, 'src_path'):
			parse_path = self.src_path
		self.shapenet_parser = ShapeNetCoreParse(parse_path)
		self.path = Path(path)
		self.len = None
		self.return_type = 'python'
		if isinstance(key, str):
			if key == '':
				key = list(self.shapenet_parser.synset_num_models.keys())
			else:
				key = [key]
		elif isinstance(key, list):
			if len(key) == 0:
				key = list(self.shapenet_parser.synset_num_models.keys())
		elif key is None:
			key = list(self.shapenet_parser.synset_num_models.keys())
		self.key = key

		self.index_mapping = self._create_index_mapping()
		self.check_data()

	def _create_index_mapping(self):
		# count amount of models
		count = 0
		for syn_name in self.key:
			if syn_name not in self.shapenet_parser.synset_num_models.keys():
				raise DatasetNotAvailableError('Failed to load dataset')
			count += self.shapenet_parser.synset_num_models[syn_name]

		index_mapping = [None] * count
		count = 0

		# map indices to (synset_id, model_id) pairs
		for key in self.key:
			synset_id = self.shapenet_parser.synset_name2id[key]
			if isinstance(self.shapenet_parser.id_subdirs[synset_id], float):
				continue
			for model_id in self.shapenet_parser.id_subdirs[synset_id]:
				index_mapping[count] = (synset_id, model_id)
				count += 1

		# shuffle indices
		random.seed(193)
		random.shuffle(index_mapping)

		return index_mapping

	def check_data(self):
		raise NotImplementedError

	def __len__(self):
		raise NotImplementedError

	def __getitem__(self, index: int):
		raise NotImplementedError

	def get_collate_fn(self):
		# creates tensors out of individual samples
		# creates torch.nn.utils.rnn.pack_sequence out of list of samples to enable use of to(device) with pytorch
		def collate_fn(batch):
			res = []
			for i in range(len(DataSourceBase.DATA_KEYS)):
				# feature is string
				if isinstance(batch[0][i], str):
					# copy needed since the array returned from np.frombuffer is not writeable if from string, causing warning
					tmp = [
						torch.from_numpy(np.frombuffer(x[i].encode('utf-8'), dtype=np.uint8).copy())
						for x in batch
					]
				# feature is data array
				elif isinstance(batch[0][i], np.ndarray):
					t = batch[0][i].dtype
					if t == 'object':
						tmp = [torch.from_numpy(np.frombuffer(x[i], dtype=np.uint8)) for x in batch]
					elif str(t).startswith('<U'):
						# copy needed since the array returned from np.frombuffer is not writeable if from string, causing warning
						tmp = [
							torch.from_numpy(np.frombuffer(x[i], dtype=np.uint8).copy())
							for x in batch
						]
					else:
						tmp = [torch.from_numpy(x[i]) for x in batch]
				# feature is already tensor
				elif isinstance(batch[0][i], torch.Tensor):
					tmp = [x[i] for x in batch]
				else:
					tmp = [torch.Tensor(x[i]) for x in batch]

				res.append(torch.nn.utils.rnn.pack_sequence(tmp, enforce_sorted=False))
			return tuple(res)

		return collate_fn

	def get_tf_output_signature(self):
		signature = []

		for _, dtype in DataSourceBase.DATA_KEYS.items():
			signature.append(tf.TensorSpec((None), dtype))

		return tuple(signature)

	def get_tf_pad_types(self):
		return tuple(
			[
				[],  # synset_id
				[],  # model_id
				[None, None],  # verts
				[None, None],  # faces
				[],  # label
			]
		)

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
	In this case, we are loading from a directory containing synset directories containing model directories with data

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	Methods
	-------
	__init__(self, path: Path | str, filename: str, key: list | str):
		creates DataSource object
	check_data(self):
		checks if we know all synsets and counts models per synset
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns model from one of specified synsets as a dict
		always acts as if return_type == python regardless of actual value
	"""

	def __init__(self, path: Path | str, key: list | str | None = None):
		super().__init__(path, key)
		self.shapenet_loader = pytorch3d.datasets.ShapeNetCore(
			data_dir=path, synsets=self.key, version=2, load_textures=False
		)

	def prepare(self):
		pass

	def check_data(self):
		try:
			# sum number of models per synset
			count = 0
			for syn_name in self.key:
				if syn_name not in self.shapenet_parser.synset_num_models.keys():
					raise DatasetNotAvailableError('Failed to load dataset')
				count += self.shapenet_parser.synset_num_models[syn_name]

			self.len = count
		except Exception:
			raise DatasetNotAvailableError

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# load the data of the mesh using pytorch3d shapenet loader
		mesh_data = self.shapenet_loader[idx]

		# convert tensors to list where necessary
		for k in mesh_data.keys():
			if isinstance(mesh_data[k], torch.Tensor):
				mesh_data[k] = mesh_data[k].tolist()

		return mesh_data


#################################
####### RAW STORAGE FORMAT ######
#################################


class DataSourceRaw(DataSourceBase):
	"""
	This class contains functions to load the data using it's original (raw) storage format.
	In this case, we are loading from a directory containing synset directories containing model directories with data

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	Methods
	-------
	check_data(self):
		checks if we know all synsets and counts models per synset
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns model from one of specified synsets as a tuple
		adapts content of tuple depending on set return type
	"""

	def __init__(self, path: Path | str, src_path: Path | str, key: list | str | None = None):
		self.src_path = src_path  # path to original data set for convenience
		super().__init__(path, key)

	def check_data(self):
		try:
			# count number of models per required synset
			count = 0
			for syn_name in self.key:
				p = self.path / self.shapenet_parser.synset_name2id[syn_name]
				if not p.is_dir():
					raise DatasetNotAvailableError('Failed to load dataset')
				for f in p.glob('*'):
					if f.is_dir():
						count += 1

			self.len = count
		except Exception:
			raise DatasetNotAvailableError

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# get path of model corresponding to <index>
		synset_id, model_id = self.index_mapping[idx]
		model_path = self.path / synset_id / model_id / 'models' / 'model_normalized.obj'

		# convert features of sample to according return type and read with according loader
		if self.return_type == 'pytorch':
			mesh = pytorch3d.io.load_obj(model_path, load_textures=False)
			return (
				synset_id,
				model_id,
				mesh[0],
				mesh[1].normals_idx,
				model_id,
			)
		elif self.return_type == 'tensorflow':
			mesh = trimesh.load(model_path, process=False, force='mesh')
			vertices = tf.convert_to_tensor(mesh.vertices, dtype=tf.float32)
			faces = tf.convert_to_tensor(mesh.faces, dtype=tf.float32)
			return (
				synset_id,
				model_id,
				vertices,
				faces,
				model_id,
			)
		else:
			mesh = trimesh.load(model_path, process=False, force='mesh')
			return (
				synset_id,
				model_id,
				np.array(mesh.vertices, dtype=np.float32),
				np.array(mesh.faces, dtype=np.float32),
				model_id,
			)


def convert_to_raw(source: DataSource, output_path: Path | str) -> DataSourceRaw:
	"""
	convert DataSource to format raw and return according DataSourceRaw object

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Raises:
		RuntimeError: raised if we try to overwrite original

	Returns:
		DataSourceRaw: reads data in raw format from output_path
	"""

	source_dir = source.path
	output_path = Path(output_path)

	if source_dir == output_path:
		raise RuntimeError('Potentially overwriting original data. Aborting')

	try:
		return DataSourceRaw(output_path, source.path, source.key)
	except DatasetNotAvailableError:
		print('copying raw dataset')

		# copy model directories of models in required synsets
		for synset_id, model_id in tqdm(source.index_mapping, total=len(source.index_mapping)):
			src_dir = source.path / synset_id / model_id
			dest_dir = output_path / synset_id / model_id
			shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

		# create empty directories for synsets without models to avoid errors
		for k in source.key:
			p = output_path / source.shapenet_parser.synset_name2id[k]
			if not p.is_dir():
				p.mkdir(parents=True, exist_ok=True)

		return DataSourceRaw(output_path, source.path, source.key)


#################################
###### HDF5 STORAGE FORMAT ######
#################################


class DataSourceHDF5(DataSourceBase):
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
		the keys of the data within the original h5 file which we consider

	H5_FILE_NAME = 'shapenet.h5'
		name of hdf5 file containing model data, relative to `path`

	Methods
	-------
	check_data(self):
		checks if we know all synsets and counts models per synset
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns model from one of specified synsets as a tuple
		adapts content of tuple depending on set return type
	"""

	H5_FILE_NAME = 'shapenet.h5'

	def __init__(self, path: Path | str, src_path: Path | str, key: list | str | None = None):
		self.src_path = src_path  # path to original data set for convenience
		super().__init__(path, key)

	def check_data(self):
		try:
			# count amount of models per required synset
			count = 0
			with h5py.File(self.path / self.H5_FILE_NAME, 'r') as fh:
				for synset in self.key:
					if self.shapenet_parser.synset_num_models[synset] > 0:
						synset_id = self.shapenet_parser.synset_name2id[synset]
						count += len(fh[synset_id].keys())

			self.len = count
		except Exception:
			raise DatasetNotAvailableError

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# get synset and model mapped to <index>
		synset_id, model_id = self.index_mapping[idx]
		mesh_data = []
		with h5py.File(self.path / self.H5_FILE_NAME, 'r') as fh:
			for key in DataSourceBase.DATA_KEYS.keys():
				data = fh[synset_id][model_id][key][...]
				# convert features of sample based on return type
				if self.return_type == 'pytorch':
					# is string
					if data.dtype == 'object':
						mesh_data.append(str(data.astype('U')))
					# list of verts or faces
					else:
						mesh_data.append(torch.from_numpy(data).to(torch.float32))
				elif self.return_type == 'tensorflow':
					mesh_data.append(
						tf.convert_to_tensor(data, dtype=DataSourceBase.DATA_KEYS[key])
					)
				else:
					mesh_data.append(data)

		return tuple(mesh_data)


def convert_to_hdf5(source: DataSource, output_path: Path | str) -> DataSourceHDF5:
	"""
	convert DataSource to format hdf5 and return according DataSourceHDF5 object
	every model and synset have their own group and in group model data is saved as data set

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Raises:
		RuntimeError: raised if we try to overwrite original

	Returns:
		DataSourceHDF5: reads data in hdf5 format from output_path
	"""

	source_dir = source.path
	output_path = Path(output_path)
	output_file = output_path / DataSourceHDF5.H5_FILE_NAME

	if source_dir == output_path:
		raise RuntimeError('Potentially overwriting original data. Aborting')

	try:
		return DataSourceHDF5(output_path, source.path, source.key)
	except DatasetNotAvailableError:
		print('converting to hdf5')

		os.makedirs(output_path, exist_ok=True)

		with h5py.File(output_file, 'w') as h5_file:
			# gather data of individual models
			for idx in tqdm(range(len(source)), total=len(source)):
				mesh_data = source[idx]
				synset_id = mesh_data['synset_id']
				model_id = mesh_data['model_id']
				syn_group = h5_file.require_group(synset_id)
				model_group = syn_group.require_group(model_id)

				# save features of models in hdf5 data sets
				for key in DataSourceBase.DATA_KEYS.keys():
					model_group.create_dataset(key, data=mesh_data[key])
				h5_file.flush()
			h5_file.close()

		return DataSourceHDF5(output_path, source.path, source.key)


#################################
###### ZARR STORAGE FORMAT ######
#################################


class DataSourceZarr(DataSourceBase):
	"""
	This class contains functions to load the data using the zarr storage format.

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	ZARR_FILE_NAME = 'shapenet.zarr'
		name of zarr file containing model data

	Methods
	-------
	check_data(self):
		checks if we know all synsets and counts models per synset
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns model from one of specified synsets as a tuple
		adapts content of tuple depending on set return type
	"""

	ZARR_FILE_NAME = 'shapenet.zarr'

	def __init__(self, path: Path | str, src_path: Path | str, key: list | str | None = None):
		self.src_path = src_path  # path to original data set for convenience
		super().__init__(path, key)

	def check_data(self):
		try:
			# count number of models per required synset
			count = 0
			data = zarr.open_group(self.path / self.ZARR_FILE_NAME, mode='r')
			for synset in self.key:
				if self.shapenet_parser.synset_num_models[synset] > 0:
					synset_id = self.shapenet_parser.synset_name2id[synset]
					synset_data = data[synset_id]
					count += len(synset_data.keys())

			self.len = count
		except zarr.errors.GroupNotFoundError:
			raise DatasetNotAvailableError
		return

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# get model and synset corresponding to ID
		synset_id, model_id = self.index_mapping[idx]

		# get model data
		obj_data = zarr.open_group(self.path / self.ZARR_FILE_NAME / synset_id / model_id, mode='r')

		mesh_data = {}
		# convert features of sample to according return type
		for k in DataSourceBase.DATA_KEYS.keys():
			data = obj_data[k][...]
			if self.return_type == 'pytorch':
				if str(data.dtype).startswith('<U'):
					mesh_data[k] = str(data)
				else:
					mesh_data[k] = torch.from_numpy(data).to(torch.float32)
			elif self.return_type == 'tensorflow':
				mesh_data[k] = tf.convert_to_tensor(data, dtype=DataSourceBase.DATA_KEYS[k])
			else:
				mesh_data[k] = data

		return tuple(mesh_data.values())


def convert_to_zarr(source: DataSource, output_path: Path | str) -> DataSourceZarr:
	"""
	convert DataSource to format zarr and return according DataSourceZarr object
	every synset and model has its own group

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceZarr: reads data in zarr format from output_path
	"""

	output_path = Path(output_path)

	try:
		return DataSourceZarr(output_path, source.path, source.key)
	except DatasetNotAvailableError:
		print('converting to zarr')

		os.makedirs(output_path, exist_ok=True)

		dest = zarr.open_group(output_path / DataSourceZarr.ZARR_FILE_NAME, mode='w')
		for idx in tqdm(range(len(source)), total=len(source)):
			# copy data of individual models to zarr groups
			mesh_data = source[idx]
			synset_id = mesh_data['synset_id']
			model_id = mesh_data['model_id']
			syn_group = dest.require_group(synset_id)
			model_group = syn_group.require_group(model_id)
			for key in DataSourceBase.DATA_KEYS.keys():
				model_group[key] = mesh_data[key]

		return DataSourceZarr(output_path, source.path, source.key)


#################################
##### PICKLE STORAGE FORMAT #####
#################################


class DataSourcePickle(DataSourceBase):
	"""
	This class contains functions to load the data using the pickle storage format.

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	PICKLE_DIR_NAME = 'shapenet.picklestore'
		directory containing individual pickle files, relative to `path`

	Methods
	-------
	check_data(self):
		checks if pickle dir exists and counts models per synset
	__len__(self):
		returns length of data set and calls check_data if not present
	__getitem__(self, index: int):
		returns model from one of specified synsets as a tuple
		adapts content of tuple depending on set return type
	"""

	PICKLE_DIR_NAME = 'shapenet.picklestore'

	def __init__(self, path: Path | str, src_path: Path | str, key: list | str | None = None):
		self.src_path = src_path  # path to original data set for convenience
		super().__init__(path, key)

	def check_data(self):
		if not (self.path / self.PICKLE_DIR_NAME).exists():
			raise DatasetNotAvailableError

		# count number of models per required synset
		count = 0
		for synset in self.key:
			if self.shapenet_parser.synset_num_models[synset] > 0:
				synset_id = self.shapenet_parser.synset_name2id[synset]
				count += len(list((self.path / self.PICKLE_DIR_NAME / synset_id).glob('*.pickle')))
		self.len = count

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# get synset and model based on <index>
		synset_id, model_id = self.index_mapping[idx]
		# retrieve model data from according pickle file
		with open(self.path / self.PICKLE_DIR_NAME / synset_id / f'{model_id}.pickle', 'rb') as f:
			mesh_data = pickle.load(f)

		if self.return_type == 'python':
			return tuple(mesh_data.values())

		# convert features of sample according to return type
		for k, t in mesh_data.items():
			if self.return_type == 'pytorch':
				if isinstance(t, str):
					continue
				else:
					mesh_data[k] = torch.Tensor(t)
			elif self.return_type == 'tensorflow':
				mesh_data[k] = tf.convert_to_tensor(t, dtype=DataSourceBase.DATA_KEYS[k])

		return tuple(mesh_data.values())


def convert_to_pickle(source: DataSource, output_path: Path | str) -> DataSourcePickle:
	"""
	convert DataSource to format pickle and return according DataSourcePickle object
	Every model is saved in an individual pickle file and there is a subdir for every synset

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourcePickle: reads data in pickle format from output_path
	"""

	output_path = Path(output_path)

	try:
		return DataSourcePickle(output_path, source.path, source.key)
	except DatasetNotAvailableError:
		print('converting to pickle')

		os.makedirs(output_path / DataSourcePickle.PICKLE_DIR_NAME, exist_ok=True)
		for idx in tqdm(range(len(source)), total=len(source)):
			# get data of model
			mesh_data = source[idx]
			synset_id = mesh_data['synset_id']
			model_id = mesh_data['model_id']

			os.makedirs(output_path / DataSourcePickle.PICKLE_DIR_NAME / synset_id, exist_ok=True)

			with open(
				output_path / DataSourcePickle.PICKLE_DIR_NAME / synset_id / f'{model_id}.pickle',
				'wb',
			) as f:
				# save model data in pickle file
				pickle_data = {}
				for key in DataSourceBase.DATA_KEYS.keys():
					pickle_data[key] = mesh_data[key]
				pickle.dump(pickle_data, f, pickle.HIGHEST_PROTOCOL)

		return DataSourcePickle(output_path, source.path, source.key)


################################
###### NPY STORAGE FORMAT ######
################################


class DataSourceNPY(DataSourceBase):
	"""
	This class contains functions to load the data using the npy storage format.

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	NPY_DIR_NAME = 'shapenet.npy'
		directory containing individual npy files, relative to `path`

	Methods
	-------
	check_data(self):
		gets len over number of npy files
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns model from one of specified synsets as a tuple
		adapts content of tuple depending on set return type
	"""

	NPY_DIR_NAME = 'shapenet.npy'

	def __init__(self, path: Path | str, src_path: Path | str, key: list | str | None = None):
		self.src_path = src_path  # path to original data set for convenience
		super().__init__(path, key)

	def check_data(self):
		if not (self.path / self.NPY_DIR_NAME / list(DataSourceBase.DATA_KEYS.keys())[0]).exists():
			raise DatasetNotAvailableError

		self.len = len(
			list(
				(self.path / self.NPY_DIR_NAME / list(DataSourceBase.DATA_KEYS.keys())[0]).glob(
					'*.npy'
				)
			)
		)

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# get model and synset corresponding to <index>
		synset_id, model_id = self.index_mapping[idx]

		mesh_data = {}
		# load data from npy files
		for key in DataSourceBase.DATA_KEYS.keys():
			mesh_data[key] = np.load(
				self.path / self.NPY_DIR_NAME / key / f'{synset_id}_{model_id}.npy'
			)

		if self.return_type == 'python':
			return tuple(mesh_data.values())

		# convert features of sample to according return type
		for k, t in mesh_data.items():
			if self.return_type == 'pytorch':
				if str(t.dtype).startswith('<U'):
					mesh_data[k] = str(t)
				else:
					mesh_data[k] = torch.Tensor(t)
			elif self.return_type == 'tensorflow':
				mesh_data[k] = tf.convert_to_tensor(t, dtype=DataSourceBase.DATA_KEYS[k])

		return tuple(mesh_data.values())


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

	try:
		return DataSourceNPY(output_path, source.path, source.key)
	except DatasetNotAvailableError:
		print('converting to npy')

		os.makedirs(output_path / DataSourceNPY.NPY_DIR_NAME, exist_ok=True)
		# create directories for features
		for key in DataSourceBase.DATA_KEYS.keys():
			os.makedirs(output_path / DataSourceNPY.NPY_DIR_NAME / key, exist_ok=True)

		for idx in tqdm(range(len(source)), total=len(source)):
			# gather relevant model data
			mesh_data = source[idx]
			synset_id = mesh_data['synset_id']
			model_id = mesh_data['model_id']

			for key in DataSourceBase.DATA_KEYS.keys():
				# save features to according npy file
				if not isinstance(mesh_data[key], str):
					mesh_data[key] = np.array(mesh_data[key]).astype(np.float32)
				np.save(
					output_path / DataSourceNPY.NPY_DIR_NAME / key / f'{synset_id}_{model_id}.npy',
					mesh_data[key],
				)

		return DataSourceNPY(output_path, source.path, source.key)


################################
###### NPZ STORAGE FORMAT ######
################################


class DataSourceNPZ(DataSourceBase):
	"""
	This class contains functions to load the data using the npz storage format.

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	NPZ_DIR_NAME = 'shapenet.npz'
		directory containing individual npz files, relative to `path`

	Methods
	-------
	check_data(self):
		sum amount of models per required synset for len
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns model from one of specified synsets as a tuple
		adapts content of tuple depending on set return type
	"""

	NPZ_DIR_NAME = 'shapenet.npz'

	def __init__(self, path: Path | str, src_path: Path | str, key: list | str | None = None):
		self.src_path = src_path  # path to original data set for convenience
		super().__init__(path, key)

	def check_data(self):
		if not (self.path / self.NPZ_DIR_NAME).exists():
			raise DatasetNotAvailableError

		# count amount of models per synset
		count = 0
		for synset in self.key:
			if self.shapenet_parser.synset_num_models[synset] > 0:
				synset_id = self.shapenet_parser.synset_name2id[synset]
				count += len(list((self.path / self.NPZ_DIR_NAME / synset_id).glob('*.npz')))
		self.len = count

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		# get synset and model corresponding to `idx`
		synset_id, model_id = self.index_mapping[idx]

		# load model data from npz files
		obj_data = np.load(self.path / self.NPZ_DIR_NAME / synset_id / f'{model_id}.npz')

		mesh_data = {}
		for key in DataSourceBase.DATA_KEYS.keys():
			mesh_data[key] = obj_data[key]

		if self.return_type == 'python':
			return tuple(mesh_data.values())

		# convert feature to according return type
		for k, t in mesh_data.items():
			if self.return_type == 'pytorch':
				if str(t.dtype).startswith('<U'):
					mesh_data[k] = str(t)
				else:
					mesh_data[k] = torch.Tensor(t)
			elif self.return_type == 'tensorflow':
				mesh_data[k] = tf.convert_to_tensor(t, dtype=DataSourceBase.DATA_KEYS[k])

		return tuple(mesh_data.values())


def convert_to_compressed_npz(source: DataSource, output_path: Path | str) -> DataSourceNPZ:
	"""
	convert DataSource to format npz and return according DataSourceNPZ object
	every sample is saved in individual file which are located in subdirs for each synset

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceNPZ: reads data in npz format from output_path
	"""

	output_path = Path(output_path)

	try:
		return DataSourceNPZ(output_path, source.path, source.key)
	except DatasetNotAvailableError:
		print('converting to compressed npz')

		os.makedirs(output_path / DataSourceNPZ.NPZ_DIR_NAME, exist_ok=True)

		# iterate over relevant models
		for idx in tqdm(range(len(source)), total=len(source)):
			# get model data
			mesh_data = source[idx]
			synset_id = mesh_data['synset_id']
			model_id = mesh_data['model_id']

			# create subdir for synset
			os.makedirs(output_path / DataSourceNPZ.NPZ_DIR_NAME / synset_id, exist_ok=True)

			# save model data to npz file
			npz_data = {}
			for key in DataSourceBase.DATA_KEYS.keys():
				npz_data[key] = mesh_data[key]
			np.savez_compressed(
				output_path / DataSourceNPZ.NPZ_DIR_NAME / synset_id / f'{model_id}.npz',
				**npz_data,
			)

		return DataSourceNPZ(output_path, source.path, source.key)


######################################
### TFRECORD_SINGLE STORAGE FORMAT ###
######################################


class DataSourceTFRecordSingle(DataSourceBase):
	"""
	This class contains functions to load the data using the tfrecord_single storage format.

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	TFRECORD_FILE_NAME = 'shapenet.tfrecord'
		name of TFRecord file containing model data

	Methods
	-------
	check_data(self):
		counts total amount of models in tfrecord file
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns model from one of specified synsets as a tuple
		adapts content of tuple depending on set return type
	"""

	TFRECORD_FILE_NAME = 'shapenet.tfrecord'

	def __init__(self, path: Path | str, src_path: Path | str, key: list | str | None = None):
		self.src_path = src_path  # path to original data set for convenience
		# set device to CPU to avoid weird errors in GPU environment during data loading
		self.device = tf.config.list_logical_devices('CPU')[0].name
		super().__init__(path, key)

	def check_data(self):
		if not (self.path / self.TFRECORD_FILE_NAME).exists():
			raise DatasetNotAvailableError

		with tf.device(self.device):
			self.dataset = tf.data.TFRecordDataset(self.path / self.TFRECORD_FILE_NAME, num_parallel_reads=tf.data.AUTOTUNE)
			self.len = sum(1 for _ in self.dataset)  # count amount of models

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

			# create parsing dictionary to get serialized tensors for all features
			parse_dic = {
				'synset_id': tf.io.FixedLenFeature([], tf.string),
				'model_id': tf.io.FixedLenFeature([], tf.string),
				'verts': tf.io.FixedLenFeature([], tf.string),
				'faces': tf.io.FixedLenFeature([], tf.string),
				'label': tf.io.FixedLenFeature([], tf.string),
			}
			mesh_data = tf.io.parse_single_example(elem, parse_dic)

			# deserialize individual features and convert according to return type
			if self.return_type == 'pytorch':
				return (
					str(mesh_data['synset_id'].numpy()),
					str(mesh_data['model_id'].numpy()),
					torch.from_numpy(
						tf.io.parse_tensor(mesh_data['verts'], out_type=tf.float32).numpy()
					),
					torch.from_numpy(
						tf.io.parse_tensor(mesh_data['faces'], out_type=tf.float32).numpy()
					),
					str(mesh_data['label'].numpy()),
				)
			elif self.return_type == 'tensorflow':
				return (
					mesh_data['synset_id'],
					mesh_data['model_id'],
					tf.io.parse_tensor(mesh_data['verts'], out_type=tf.float32),
					tf.io.parse_tensor(mesh_data['faces'], out_type=tf.float32),
					mesh_data['label'],
				)
			else:
				return (
					str(mesh_data['synset_id'].numpy()),
					str(mesh_data['model_id'].numpy()),
					tf.io.parse_tensor(mesh_data['verts'], out_type=tf.float32).numpy(),
					tf.io.parse_tensor(mesh_data['faces'], out_type=tf.float32).numpy(),
					str(mesh_data['label'].numpy()),
				)


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
		return DataSourceTFRecordSingle(output_path, source.path, source.key)
	except DatasetNotAvailableError:
		print('converting to tfrecord single')

		os.makedirs(output_path, exist_ok=True)
		fname_out = (
			(output_path / DataSourceTFRecordSingle.TFRECORD_FILE_NAME).absolute().as_posix()
		)

		with tf.io.TFRecordWriter(fname_out) as tf_writer:
			# iterate over relevant models
			for idx in tqdm(range(len(source)), total=len(source)):
				mesh_data = source[idx]
				# accordingly serialize data of model
				tf_mesh_data = {
					'synset_id': tf.train.Feature(
						bytes_list=tf.train.BytesList(
							value=[mesh_data['synset_id'].encode('utf-8')]
						)
					),
					'model_id': tf.train.Feature(
						bytes_list=tf.train.BytesList(value=[mesh_data['model_id'].encode('utf-8')])
					),
					'verts': tf.train.Feature(
						bytes_list=tf.train.BytesList(
							value=[
								tf.io.serialize_tensor(
									tf.convert_to_tensor(mesh_data['verts'])
								).numpy()
							]
						)
					),
					'faces': tf.train.Feature(
						bytes_list=tf.train.BytesList(
							value=[
								tf.io.serialize_tensor(
									tf.cast(tf.convert_to_tensor(mesh_data['faces']), tf.float32)
								).numpy()
							]
						)
					),
					'label': tf.train.Feature(
						bytes_list=tf.train.BytesList(value=[mesh_data['label'].encode('utf-8')])
					),
				}
				# save model data to file
				record_bytes = tf.train.Example(
					features=tf.train.Features(feature=tf_mesh_data)
				).SerializeToString()
				tf_writer.write(record_bytes)

		return DataSourceTFRecordSingle(output_path, source.path, source.key)


#####################################
### TFRECORD_MULTI STORAGE FORMAT ###
#####################################


class DataSourceTFRecordMulti(DataSourceBase):
	"""
	This class contains functions to load the data using the tfrecord_multi storage format.

	Attributes
	----------
	path : Path | str
		The base path of the dataset
	key : list | str | None
		The subset of classes we will consider (if none or empty list/string we use all)

	TFRECORD_DIR_NAME = 'shapenet.tfrecord'
		directory relative to `path` containing tfrecord files

	Methods
	-------
	check_data(self):
		checks if we know all synsets and counts models per synset
	__len__(self):
		returns amount of samples
	__getitem__(self, index: int):
		returns model from one of specified synsets as a tuple
		adapts content of tuple depending on set return type
	"""

	TFRECORD_DIR_NAME = 'shapenet.tfrecord'

	def __init__(self, path: Path | str, src_path: Path | str, key: list | str | None = None):
		self.src_path = src_path  # path to original data set for convenience
		# set device to CPU to avoid weird errors in GPU environment during data loading
		self.device = tf.config.list_logical_devices('CPU')[0].name
		super().__init__(path, key)

	def check_data(self):
		if not (self.path / self.TFRECORD_DIR_NAME).exists():
			raise DatasetNotAvailableError

		# count amount of models per synset
		count = 0
		for synset in self.key:
			synset_id = self.shapenet_parser.synset_name2id[synset]
			count += len(list((self.path / self.TFRECORD_DIR_NAME / synset_id).glob('*.tfrecord')))
		self.len = count

	def __len__(self):
		if self.len is None:
			self.check_data()
		return self.len

	def __getitem__(self, index: int):
		idx = index % self.len

		with tf.device(self.device):
			# read according sample
			synset_id, model_id = self.index_mapping[idx]
			dataset = tf.data.TFRecordDataset(
				self.path / self.TFRECORD_DIR_NAME / synset_id / f'{model_id}.tfrecord',
				num_parallel_reads=tf.data.AUTOTUNE,
			)
			elem = None
			for data in dataset.take(1):
				elem = data

			# create parsing dictionary to get serialized tensors for all features
			parse_dic = {
				'synset_id': tf.io.FixedLenFeature([], tf.string),
				'model_id': tf.io.FixedLenFeature([], tf.string),
				'verts': tf.io.FixedLenFeature([], tf.string),
				'faces': tf.io.FixedLenFeature([], tf.string),
				'label': tf.io.FixedLenFeature([], tf.string),
			}
			mesh_data = tf.io.parse_single_example(elem, parse_dic)

			# deserialize features and convert according to set return_type
			if self.return_type == 'pytorch':
				return (
					str(mesh_data['synset_id'].numpy()),
					str(mesh_data['model_id'].numpy()),
					torch.from_numpy(
						tf.io.parse_tensor(mesh_data['verts'], out_type=tf.float32).numpy()
					),
					torch.from_numpy(
						tf.io.parse_tensor(mesh_data['faces'], out_type=tf.float32).numpy()
					),
					str(mesh_data['label'].numpy()),
				)
			elif self.return_type == 'tensorflow':
				return (
					mesh_data['synset_id'],
					mesh_data['model_id'],
					tf.io.parse_tensor(mesh_data['verts'], out_type=tf.float32),
					tf.io.parse_tensor(mesh_data['faces'], out_type=tf.float32),
					mesh_data['label'],
				)
			else:
				return (
					str(mesh_data['synset_id'].numpy()),
					str(mesh_data['model_id'].numpy()),
					tf.io.parse_tensor(mesh_data['verts'], out_type=tf.float32).numpy(),
					tf.io.parse_tensor(mesh_data['faces'], out_type=tf.float32).numpy(),
					str(mesh_data['label'].numpy()),
				)


def convert_to_tfrecords_multi(
	source: DataSource, output_path: Path | str
) -> DataSourceTFRecordMulti:
	"""
	convert DataSource to format tfrecord_multi and return according DataSourceTFRecordMulti object
	all samples are saved in individual TFRecord files in subdirectories for the synsets

	Args:
		source (DataSource): object containing data for conversion
		output_path (Path | str): location to save conversion

	Returns:
		DataSourceTFRecordMulti: reads data in tfrecord_multi format from output_path
	"""

	output_path = Path(output_path)

	try:
		return DataSourceTFRecordMulti(output_path, source.path, source.key)
	except DatasetNotAvailableError:
		print('converting to tfrecord multi')

		os.makedirs(output_path / DataSourceTFRecordMulti.TFRECORD_DIR_NAME, exist_ok=True)
		for idx in tqdm(range(len(source)), total=len(source)):
			mesh_data = source[idx]
			synset_id = mesh_data['synset_id']
			model_id = mesh_data['model_id']

			# create subdirectory for synset
			os.makedirs(
				output_path / DataSourceTFRecordMulti.TFRECORD_DIR_NAME / synset_id, exist_ok=True
			)
			fname_out = (
				(
					output_path
					/ DataSourceTFRecordMulti.TFRECORD_DIR_NAME
					/ synset_id
					/ f'{model_id}.tfrecord'
				)
				.absolute()
				.as_posix()
			)
			with tf.io.TFRecordWriter(fname_out) as tf_writer:
				# serialize model data
				tf_mesh_data = {
					'synset_id': tf.train.Feature(
						bytes_list=tf.train.BytesList(
							value=[mesh_data['synset_id'].encode('utf-8')]
						)
					),
					'model_id': tf.train.Feature(
						bytes_list=tf.train.BytesList(value=[mesh_data['model_id'].encode('utf-8')])
					),
					'verts': tf.train.Feature(
						bytes_list=tf.train.BytesList(
							value=[
								tf.io.serialize_tensor(
									tf.convert_to_tensor(mesh_data['verts'])
								).numpy()
							]
						)
					),
					'faces': tf.train.Feature(
						bytes_list=tf.train.BytesList(
							value=[
								tf.io.serialize_tensor(
									tf.cast(tf.convert_to_tensor(mesh_data['faces']), tf.float32)
								).numpy()
							]
						)
					),
					'label': tf.train.Feature(
						bytes_list=tf.train.BytesList(value=[mesh_data['label'].encode('utf-8')])
					),
				}
				# save serialized data to individual tfrecord file
				record_bytes = tf.train.Example(
					features=tf.train.Features(feature=tf_mesh_data)
				).SerializeToString()
				tf_writer.write(record_bytes)

		return DataSourceTFRecordMulti(output_path, source.path, source.key)


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
	file_path = npy.path / npy.NPY_DIR_NAME

	# list of all relevant npy files
	file_list = [f'{synset_id}_{model_id}.npy' for synset_id, model_id in source.index_mapping]

	return (file_path, file_list)


@pipeline_def
def dali_npy_pipeline(file_path: str | Path, file_list: list, device: str):
	"""
	DALI pipeline definition using numpy reader
	only reads faces and vertices, not model or synset id

	Args:
		file_path (str | Path): root directory of npy subfolders
		file_list (list): list of numpy files in subfolders
		device (str): device to load samples to
	"""

	# reader for vertices of mesh
	verts = fn.readers.numpy(
		device=device,
		file_root=file_path / 'verts',
		files=file_list,
		name='readerverts',
		random_shuffle=True,
		seed=193,
		lazy_init=True,
	)

	# reader for faces of mesh
	faces = fn.readers.numpy(
		device=device,
		file_root=file_path / 'faces',
		files=file_list,
		name='readerfaces',
		random_shuffle=True,
		seed=193,
		lazy_init=True,
	)

	# pad data for batching
	data = [
		fn.pad(verts[0], device=device),
		fn.pad(faces[0], device=device),
	]

	return tuple(data)


#####################################
### TFRECORD_SINGLE DALI PIPELINE ###
#####################################


def dali_tfrecord_single_prepare(source: DataSource, output_path: str | Path):
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

	# make sure conversion is done
	file_path = tf_single.path / tf_single.TFRECORD_FILE_NAME

	# create index file required by tfrecord reader
	idx_path = tf_single.path / 'idx_files'
	if not idx_path.is_dir():
		idx_path.mkdir(parents=True, exist_ok=True)

	idx_path = idx_path / 'shapenet.idx'
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
	# string since tensors are serialized
	keys = ['verts', 'faces']
	features = {}
	for k in DataSourceBase.DATA_KEYS:
		features[k] = tfrec.FixedLenFeature([], tfrec.string, '')

	inputs = fn.readers.tfrecord(
		path=file_path,
		index_path=idx_path,
		features=features,
		random_shuffle=True,
		seed=193,
		lazy_init=True,
		name='readerverts',  # little workaround for length
	)

	data = []
	for k in keys:
		# since tf reader does not suppport gpu device omitted
		data.append(fn.pad(inputs[k]))
	return tuple(data)


##################################
#### AVAILABLE DALI PIPELINES ####
##################################


def get_available_pipelines(keys: list):
	pipelines = {
		'npy': {
			'preparation': dali_npy_prepare,
			'pipeline': dali_npy_pipeline,
			'labels': ['verts', 'faces'],
			'pytorch_args': {},
			'tensorflow_args': {'shapes': [(), ()], 'dtypes': [tf.float32, tf.float32]},
		},
		'tfrecord_single': {
			'preparation': dali_tfrecord_single_prepare,
			'pipeline': dali_tfrecord_single_pipeline,
			'labels': ['verts', 'faces'],
			'pytorch_args': {},
			'tensorflow_args': {'shapes': [(), ()], 'dtypes': [tf.float32, tf.float32]},
		},
	}

	return pipelines
