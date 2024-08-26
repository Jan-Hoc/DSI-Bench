# DSI-Bench

## Project Introduction
DSI-Bench is a benchmarking framework to facilitate the performance measurements of data storage and ingestion for data science applications. It enables the comparison of different metrics, namely runtime, CPU utilization and RAM utilization, for different combinations of data sets, consumers and storage formats. See [Implemented Features](#implementated-features) for the details.

These comparisons allow you to simply choose the best combination of storage format, consumer and data set for your unique situation.

DSI-Bench's [structure](#framework-structure) easily enables you to add classes for your own data sets, consumers and storage formats, which are not implemented out of the box. These possibilities are described in [Extending the Benchmark](#extending-the-benchmark). 

For more detailed information to this project, including sample results, please refer to the corresponding [thesis](paper/thesis.pdf). The rest of the README also contains an extensive guide.

To view further benchmark results, execute the script `paper/paper_results/create_graphs.py`. It generates graphs from the results used in the [thesis](paper/thesis.pdf) as well as from some further suites. The graphs can then be found in the corresponding subdirectories of `paper/paper_results`.

<hr>

- [DSI-Bench](#dsi-bench)
	- [Project Introduction](#project-introduction)
	- [Quickstart](#quickstart)
		- [Installation](#installation)
		- [Docker](#docker)
		- [Execution](#execution)
	- [Configuration](#configuration)
		- [Data Directory](#data-directory)
		- [Config Files](#config-files)
	- [Implementated Features](#implementated-features)
		- [Storage Formats](#storage-formats)
		- [Consumers](#consumers)
		- [Data Sets](#data-sets)
		- [Metrics](#metrics)
			- [System Information](#system-information)
			- [Size of converted Data Sets](#size-of-converted-data-sets)
			- [Run Results](#run-results)
			- [Graphs](#graphs)
	- [Framework Structure](#framework-structure)
		- [General Structure](#general-structure)
			- [Consumers](#consumers-1)
			- [Data Sets](#data-sets-1)
		- [Execution of the Benchmark](#execution-of-the-benchmark)
		- [Extending the Benchmark](#extending-the-benchmark)
			- [Adding a Data Set](#adding-a-data-set)
			- [Adding a Storage Format](#adding-a-storage-format)
			- [Adding a Consumer](#adding-a-consumer)

## Quickstart

### Installation

Install the latest [miniconda](https://docs.conda.io/en/latest/miniconda.html) and then the required dependencies with the script `environment/install-env.sh`. It will create a conda environment containing all dependencies. You may pass an argument to name the environment, else it will be named `benchmark`.

```
./environment/install-env.sh <environment_name>
```

### Docker

Alternatively you can also run a docker container with all the dependencies. There are three images, which can be found on [DockerHub](https://hub.docker.com/r/majorhph/dsi-bench), including more information. 

There is one basic image `latest` containing the current state of this repository as well as an installation of conda. This image is built everytime the `main` branch of the [official repository](https://github.com/Jan-Hoc/DSI-Bench) is updated. The conda environment is not built yet to reduce the image size.

The other two tags are `standard` and `runai`. Standard is equivalent to `latest`, but has already built the conda environment, which is available under the name `benchmark`. The `runai` image contains some further extensions for the development on the [SDSC Run:ai infrastructure](https://sdsc.run.ai). These two images are not integrated into the GitHub pipeline, as they are quite large. They can be built using one of the two following commands, depending on the desired version. Setting `build=0` will not build the conda environment, while `build=1` does install it.

```bash
docker build --build-arg version=standard --build-arg build=1 . -f Dockerfile
docker build --build-arg version=runai --build-arg build=1 . -f Dockerfile
```

To pull and run the docker image, execute the following commands. You should be logged into the container as `dev`.

```bash
docker pull majorhph/dsi-bench:latest
docker run -it majorhph/dsi-bench:latest
```

If you want to keep the container running when detatching (e.g. because a benchmark is running), use `Ctrl+p, Ctrl+q` instead of `Ctrl+d`.


### Execution

To run the benchmark execute the command:
```
python benchmark/benchmark.py <flags>
```

For the flags the list of possible values is also the list of default values. Every combination of `suites`, `converter_names`, `consumers`, `devices`, and `return_types` is benchmarked.

| Flags | Possible Values | Description |
| :---: | :---: | :---: |
| `suites` | `TEST`, `OADAT`, `ICON`, `ImageNet`, `ShapeNet` | The `suite_name` given in the configuration files, which should be benchmarked. |
| `converter_names` | `raw`, `hdf5`, `zarr`, `pickle`, `npy`, `npz`, `tfrecord_single`, `tfrecord_multi` | The storage formats to which the datasets should be converted to for the benchmark. |
| `consumers` | `python`, `tensorflow`, `tensorflow_sequence`, `pytorch`, `dali_pytorch`, `dali_tensorflow` | The consumers with which the data should be read during the benchmark. <br> The `DALI` consumers are only combatible with `npy`, `tfrecord_single` and for `ImageNet` `raw`. |
| `devices` | `cpu`, `gpu` | To which device the tensors should be loaded to. The option `gpu` is only available if `python -c "import torch; print(torch.cuda.is_available())"` prints `True`. |
| `return_types` | `python`, `pytorch`, `tensorflow` | What type of object the dataset's `__getitem__` should return for the dataloader. Either a Python native object like a tuple or dict, or a PyTorch/TensorFlow tensor. |
| `name_prefix` | `*` | A string which is added as a prefix to the dataset name, forming the name of the benchmark for that dataset configuration. |
| `rounds` | `1`, ..., `999` | The amount of rounds, which every run of the benchmarked configurations should be repeated for. A higher amount gives more reliable results but also takes longer to execute. |
| `delete_conversions` / `keep_conversions` | N/A | These flags enable you to keep or delete the converted data sets in the temporary directory. After each benchmark suite has completed, the corresponding conversion in `TMP_DATA_DIRECTORY` is deleted. The default is `delete_conversions` |

## Configuration
### Data Directory

By default the benchmarks persist files to disk for each dataset. These are stored on the disk to which you downloaded the repository, i.e. in the `tmp` folder this project. If you don't have enough space on this disk the benchmark might fail. You can either clone the repository to a different disk or change the location of the persisted data by creating a `.env` file or by copying the `.env.sample` (with the command `cp .env.sample .env`). We recommend to change the value of `TMP_DATA_DIRECTORY` to a directory with enough space. Be aware that having the data on a different disk might have an impact on the benchmark performance.
```
TMP_DATA_DIRECTORY="/your/folder/with/space/"
```

### Config Files

You can configure one or several runs/different benchmark versions using YAML files. One for each available dataset can be found in the directory `configs`. The following keys need to be set for all dataset types:

| Key | Meaning |
| :---: | :---: |
| `suite_name` | This is the name, that you need to choose with the flag `suites` if you don't want to run all runs. It must be unique under the available configs. |
| `dataset_type` | This is the name of the file which holds the corresponding dataset class (e.g. `imagenet`). |
| `path` | This is the absolute path, where the original dataset is located. |
| `batch_size` | This is the used batch size used for data loading. |
| `num_threads` | The amount of threads, which should be used for data loading. |

Any parameters other than `path`, that the classes of `dataset_type` use in the constructor also need to be present in the YAML config. For the currently implemented datasets these are the following.

| Key | Applicable Data Set | Meaning |
| :---: | :---: | :---: |
| `filename` | Test, OADAT, ICON | The name of the actual HDF5 file containing the features, relative to `path`. <br> Ignored for the Test data set but present for demonstration purposes. |
| `key` | Test, OADAT, ICON, ShapeNet | For OADAT and ICON the list of keys/features, which should be used for benchmarking. <br> For ShapeNet this is a list of category names used for the benchmark (e.g. `bicycle`) <br> Ignored for the Test data set but present for demonstration purposes. |
| `amount` | ImageNet | The amount of images, which should be used for the benchmark. They need to be directly located in `path`, which is why it is recommended to pass the location of the validation data set in `path`. That directly contains the images. |

If you create a new configuration file for an existing data set, you can add the unique value you used for `suite_name` to `default_suites` in `benchmarks/benchmarks.py`. Then it will be used as a default option, else it will only be executed if you specify it using the CLI argument `suites`.

## Implementated Features
### Storage Formats

The following storage formats are implemented by all current data set classes. The names are the ones returned as keys of the dictionary by the function `get_available_converters` in the Python file of the corresponding data set. More to this is in the section about the benchmark structure.

- `raw`: The original storage format of the data set. For the different data sets these are the following:
  -  Test: directly returning entries from an in memory array
  -  OADAT: HDF5
  -  ICON: HDF5
  -  ImageNet (ILSVRC 2012): JPEG
  -  ShapeNet: Meshes as OBJ files
-  `hdf5`
-  `zarr`
-  `pickle`: Pickled using `pickle.HIGHEST_PROTOCOL`.
-  `npy`
-  `npz`: Saved compressed using `np.savez_compressed`.
-  `tfrecord_single`: The features used in the benchmark are all saved in one single TFRecord file. They're serialized during storage and deserialized during reading.
-  `tfrecord_multi`: every feature is saved in an individual TFRecord file. They're serialized during storage and deserialized during reading.

### Consumers
The consumers will be referenced with the corresponding CLI argument, with which they can be used. The class names in parenthesis are as found in the file `loader-benchmark/consumers.py`. They may build on other data loader classes, which supply the elements to the consumer using the according library. More will be mentioned on this later in the section about the benchmark structure.

- `python` (`PythonConsumer`): Consumes the data without any additional deep learning frameworks as PyTorch or TensorFlow. Implements multi-threading using `concurrent.futures.ThreadPoolExecutor`. This is used instead of `concurrent.futures.ProcessPoolExecutor` since it is an I/O and not CPU heavy workload.
- `tensorflow` (`TensorflowConsumer`): Uses a data loader based on the TensorFlow data API. If necessary the data is padded to allow for batching. The data is consumed using an iterator based on a `tf.data.Dataset`
- `tensorflow_sequence` (`TensorflowSequenceConsumer`): This uses a data loader based on `tf.keras.utils.Sequence`
- `pytorch` (`PytorchConsumer`): The data loader for the PyTorch is `torch.utils.data.DataLoader`. If passed it uses a collate function to enable batching, similar to the padding of the TensorFlow consumer.
- `dali_pytorch` (`DALIPytorchConsumer`): Takes the DALI pipelines (defined in the data set files) and iterates through it randomly using `NVIDIA.dali.plugin.pytorch.DALIGenericIterator`
- `dali_tensorflow` (`DALITensorflowConsumer`): Similar to `DALIPytorchConsumer` but uses `NVIDIA.dali.plugin.tf.DALIIterator` instead.

### Data Sets

- Test (`dsi_bench/datasets/test.py`): This is simply a data set intended to check if all functions work and to demonstrate how to write the different converters. It generates an array with 10'000 integers which are treated as the sample data. For the DALI loaders this provides two pipelines, one based on `npy` and one on the `tfrecord_single` format.
- OADAT (`dsi_bench/datasets/oadat.py`): Medical data stored in an HDF5 file. For the DALI loaders this provides two pipelines, one based on `npy` and one on the `tfrecord_single` format.
- ICON (`dsi_bench/datasets/icon.py`): Climate simulation data stored in an HDF5 file. For the DALI loaders this provides two pipelines, one based on `npy` and one on the `tfrecord_single` format.
- ImageNet (ILSVRC 2012) (`dsi_bench/datasets/imagenet.py`): A data set providing images in JPEG format for image classification. For the DALI loaders this provides three pipelines, one based on `raw`, so the JPEG images, one based on `npy` and one on the `tfrecord_single` format.
- ShapeNet v2 (`dsi_bench/datasets/shapenet.py`): A data set providing labeled meshes as OBJ files. For the DALI loaders this provides two pipelines, one based on `npy` and one on the `tfrecord_single` format.

### Metrics

There are four main metrics, which are gathered during the benchmark. The first three measure the system load and are collected by [PySysLoadBench](https://github.com/Jan-Hoc/PySysLoadBench).

- CPU load `cpu` (in percent where 100% is equivalet to one core being fully utilized)
- RAM utilization `ram` (in MiB for the graphs as well as bytes for the raw results)
- Time `time` needed to consume the data set (or subset specified by the config file)
- Space `used_diskspace` needed by the converted data sets (in MiB)

The metrics `cpu` and `ram` are collected every 0.05 seconds and the value of the process used for data loading as well as all child processes are accounted for and summed up. For more details refer to [PySysLoadBench](https://github.com/Jan-Hoc/PySysLoadBench).

Every benchmark defined by a configuration file is regarded seperately and there is a folder `results/<name_prefix><suite_name>` containing the results. For one there is a JSON file `raw_results.json`, containing some general system information, the size of the converted data (sub-)sets and the results of the different runs.

#### System Information

The system information can be found under the key `system_information` and contains the following subkeys

- `python_version`: Python version
- `platform`: Platform
- `operating_system`: Operating system
- `host_name`: Host name
- `cpu`: CPU model
- `gpu`: GPU model (empty if none available)
- `ram`: Amount of available RAM

#### Size of converted Data Sets

This information is found under the key `used_diskspace` and the subkeys are the arguments given to the CLI option `converter_names` (or all possible if none given). The number is the diskspace used in MiB.

Beware that this does not refer to the whole dataset, since only the portion used in the data set is converted. Since for the `raw` option the needed data is copied as well to `TMP_DATA_DIRECTORY`, this also only accounts for the used data. Therefore, this metric is rather suited for qualitative comparisons among the different storage formats and not quantitatively with the original full data set.

#### Run Results

The run results are found under the key `run_results` and have the following structure:
- `<converter_name> <consumer> <device> <return_type>` (due to the different structure with the DALI consumers the return type part of the key is omitted in these cases)
  - `<metric>`: either `cpu` (value in percent with 100% = 1 CPU core fully utilized), `ram` (in byte) or `time` (in seconds).
    - [`0`, `n-1`] where `n` is the argument of `rounds`. The numerically indexed fields hold the stats of the  corresponding individual run. This is not available for the metric `time`.
      - `max`: Maximum recorded value of metric over run.
      - `mean`: Mean value of metric over round.
      - `stddev`: Standard deviation of recorded values of metric.
      - `25`: 25th percentile of data.
      - `50`: 50th percentile of data.
      - `75`: 75th percentile of data.
      - `90`: 90th percentile of data.
      - `95`: 95th percentile of data.
      - `99`: 99th percentile of data. 
    - `raw`: Only available for the metric `time`. List of raw measurements of the runtime in seconds of the different rounds.
    - `total`: Holds the stats of summarized over all rounds (based on `mean` of rounds).
      - `max`: Maximum mean of metric over rounds.
      - `mean`: Mean of mean of metric over all rounds.
      - `stddev`: Standard deviation over the mean of the rounds (or over `raw` entries in case of metric `time`).
      - `25`: 25th percentile of data.
      - `50`: 50th percentile of data.
      - `75`: 75th percentile of data.
      - `90`: 90th percentile of data.
      - `95`: 95th percentile of data.
      - `99`: 99th percentile of data.

#### Graphs

There are also graphs which are generated to visually compare the metrics. All the paths following are relative to the results directory of the corresponding benchmark. The graphs contain the value `total/mean` of the raw results and also indicate the standard deviation.

The graph `<name_prefix><suite_name>_diskspace.png` compares the metric `used_diskspace` (in MiB) of the different used storage formats.

The bar graphs `converter_graphs/<metric>/<converter_name>_<metric>_<device>_<return_type>.png` compare the metric of the different consumers with each other when `<converter_name>` is used as a storage format. If the DALI consumers are used, the result is added to all used return types to facilitate comparison. If only DALI consumers are used `<return_type>=default` instead of one of the possible values of the CLI argument.

The bar graphs `consumer_graphs/<metric>/<consumer>_<metric>_<device>_<return_type>.png` compare the metric of the different storage formats with each other when `<consumer>` is used as a consumer. If the DALI consumers are used, the result is added to the graphs for all used return types to facilitate comparison. If only DALI consumers are used `<return_type>=default` instead of one of the possible values of the CLI argument.

The graphs `summary_graphs/<metric>/<metric>_<device>_<return_type>.png` are heat maps to quickly compare all combinations of consumers and storage formats on one sight. To increase the readability the concrete numbers are omitted, however there is a labeled color scale. If the DALI consumers are used, the result is added to the graphs for all used return types to facilitate comparison. If only DALI consumers are used `<return_type>=default` instead of one of the possible values of the CLI argument. For a more detailed comparison there is also a labeled version under `summary_graphs/<metric>/<metric>_<device>_<return_type>_label.png` for every summary graph.

## Framework Structure
### General Structure

The code in `benchmarks/benchmarks.py` ties the different components together and acts as the "entry point". it reads the CLI arguments, config files etc. and starts the various benchmarks, which are given through the different valid combinations of CLI arguments. It handles things like passing the correct arguments to the various consumers for you. For completeness and potential trouble shooting these things are  still explained later.

The directory `dsi_bench` contains the other needed code. For one there are the files `illustrator.py`, responsible for creating the graphs, as well as `parse.py`, which reads the YAML config files. Then `consumers.py` contains the code for the different consumers. More details to the consumers can be found in the following subsection. The subdirectory `datasets` contains a python file for each available data set (`test.py`, `oadat.py`, `icon.py`, `imagenet.py`, `shapenet.py`). More to this also in a following subsection.

The directory `data` contains soft links to the different data sets to facilitate access during developement. When the benchmark is run however, only the `path` in the config file is relevant, which should note the direct and absolute path and not go over the soft link.

The directory `results` contains the results, as mentioned in a previous section.

The directory `configs` contains the config files, which where mentioned in a previous section to configuration.


#### Consumers

There is a consumer class for every possible option of the CLI argument `consumers`. These posess an implementation of the `__call__` function to consume the data set. Some of these consumers also rely on an additional consumer class, which provides an implementation of `__getitem__` to provide data using the according framework.

All consumers take the amount of threads (`num_threads`) to be used, as well as the batch size (`batch_size`) as an argument. These are the values defined in the config files.

For the "normal" (not DALI) consumers, they take an instance of the class `DataSourceBase`, which is explained in the next section. This will be the data set they consume. 

The `TensorflowConsumer` additionally requires that the data set class defines the functions `get_tf_output_signature` and `get_tf_pad_types`. The former needs to return an instance of `tf.TensorSpec` specifying the output signature, which is returned by the data set classes `__getitem__`. The return value of the latter is passed to the argument `padded_shapes` of `tf.data.Dataset.padded_batch`. This is only necessairy if not all elements returned by `__getitem__` have the same size. Without padding batching would not be possible in this case. If all elements returned by `__getitem__` have the same size, `get_tf_pad_types` may return `None` and `padded_batch` is not called.

The `PytorchConsumer` requires the data set class to implement `get_collate_fn`, which returns a collate function, defining how to collate individual elements returned by `__getitem__` into a single batch. If all elements returned by `__getitem__` have the same dimension, this is not required and `get_collate_fn` may return `None`.

The DALI consumers (`DALIPytorchConsumer`, `DALITensorflowConsumer`) do not take an instance of `DataSourceBase`, but a NVIDIA DALI pipeline. More on how to define this for every data set in the next section. They both each also take the argument `device` (`cpu` or `gpu`). Further, the `DALIPytorchConsumer` requires the argument `labels`, which is a list of strings on containing labels for the different features contained in the sample returned by the pipeline. If none are given and the corresponding `DataSourceBase` has an attribute `key`, that list is chosen. To infer the length of the data sets, the readers in the pipeline must be named. If there is one for every feature `<label[i]>`, they need to be named `reader<label[i]>`. If there is only one reader it needs to be named `reader<label[0]>`. The `DALITensorflowConsumer` requires two additional arguments, `shapes` and `dtypes`. These, similar to the value returned by `get_tf_output_signature`, define the shape and data type of the sample returned by the pipeline. Both consumers also take the argument `pipe_args`, which may be a tuple of additional arguments which need to be passed to the pipeline. More on where to define all this and the pipelines is in the next section.

If the returned data is a tensor, it will be loaded to the according `device` during the consumption of the data set. Be aware that that is why the combination `device=gpu`, `PythonConsumer` is skipped, since we dont necessarily have a tensor here in the end and can therefore not load it into the GPU memory.

#### Data Sets

For every data set there is an according file in the subdirectory `datasets`. Each data set file must contain a class `DataSourceBase`, who's initializer takes the `path` as well as any other arguments defined in the according configuration file. Further the functions `__len__`, `__getitem__` and `check_data` must be defined (but may raise `NotImplementedError`, since this class is more used as a parent class and isn't actually used for benchmarking). As mentioned before `get_tf_output_signature`, `get_tf_pad_types` and `get_collate_fn` must be implemented here too. The last two may return `None` if the `__getitem__` functions always return elements of the same dimensions. The functions `get_available_devices`, `get_available_return_types` also need to be implemented, which return the implemented values and return types by the data set classes, which may be passed to `set_device` and `set_return_type`, which alter the behaviour of the data set class. 

In the `__getitem__` function of the dataset classes the returned value changes depending on the return type. If it is `python` a Python native object like a `tuple` or `dict` is returned. In case it is `pytorch` or `tensorflow`  corresponding tensor is already returned. This is to check if it makes a difference if we already create a tensor at this stage and don't only leave it to the consumer. The `PytorchConsumer` is not compatible with the return type `tensorflow`, so this combination is skipped. No matter the return type here, the `PytorchConsumer` and `DALIPytorchConsumes` will return a PyTorch tensor and the `TensorflowConsumer`, `TensorflowSequenceConsumer` and `DALITensorflowConsumer` will return a TensorFlow tensor. The `PythonConsumer` will return whatever is given.

All of the following mentioned classes must implement `DataSourceBase`.

There also must be a class `DataSource`, which implements `__len__`, `__getitem__` and `check_data`. `check_data` is supposed to do a rudimentary check of the data and raise `DatasetNotAvailableError` (defined as a child class of `Exception` at the start of the data set files) if not all data needed for the benchmark is available (e.g. a key in the HDF5 file is missing). The idea here is that `__getitem__` returns the original data located at `path`. It is mainly used for the conversion to other formats and not acbual benchmarking. That's why it is recommended that the `__getitem__` function always behaves as if `device=cpu` and `return_type=python`, as it facilitates the conversion.

Then for every by this data set implemented storage format we need a function converting the original data set and a class returning the according data. An examble would be `DataSourceHDF5` and `convert_to_hdf5`. The conversion function takes an instance of `DataSource` and converts all the data required by the benchmark. It also returns an instance of the corresponding class (e.g. `DataSourceHDF5`). The conversion function also bakes an argument `output_path`, which is where to save the converted data. This is `<TMP_DATA_DIRECTORY>/<suite_name>/<converter_name>`.

The dataset file also contains a function `get_available_converters` returning a dictionary, mapping `converter_names` to the conversion functions. If a data set does not implement a certain storage format conversion, just omit that key. An example for a data set implementing all storage formats would be

```python
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
```

As mentioned earlier the DALI consumers need pipelines. Depending on the different data formats supported by DALI and implemented in the data set class we have various pipeline definitions. Please refer to the [official NVIDIA DALI documentation](https://docs.NVIDIA.com/deeplearning/dali/user-guide/docs/index.html) for more details. In most cases there is a defintion for a `npy` and `tfrecord_single` pipeline definition. These may take arbitrary arguments, so for every pipeline you define there is a another preparation function (e.g. `dali_npy_prepare`). These make sure the needed data is present and return the arguments needed by the corresponding pipeline. So the `DALIPytorchConsumer` can infer the length of the data set, each reader of the pipeline needs to be named. If there is a reader for every feature `<label[i]>`, they need to be named `reader<label[i]>`. If there is only one reader it needs to be named `reader<label[0]>`. For an example take a look at the following code, taken from the Test data set. It also shows the two arguments the preparation function must take, as well as the naming of the readers in the pipeline definition.

```python
def dali_npy_prepare(source: DataSource, output_path: str | Path) -> tuple:
	# make sure data is converted
	npy = get_available_converters()['npy'](source, output_path)

	# directory containing npy files
	file_path = npy.path / npy.filename

	# list of all relevant npy files
	file_list = [str(i) + '.npy' for i in range(len(npy))]

	return (file_path, file_list, npy.key)


@pipeline_def
def dali_npy_pipeline(file_path: str | Path, file_list: list, keys: list, device: str):
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
```

Similar to `get_available_converters` there is an equivalent `get_available_pipelines`, which maps the supported storage formats to the corresponding preparation functions and pipeline definitions. Further it specifies the labels and the arguments which are only passed to the `DALIPytorchConsumer` or `DALITensorflowConsumer`. It could look like this:

```python
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
```

Since the keys are variable and they determine `labels` and the `tensorflow_args` they are passed to the `get_available_pipelines` function whenever they are a part of the constructor of the corresponding `DataSourceBase`. For other data sets (like ImageNet) this function mustn't take any arguments.

### Execution of the Benchmark

For all valid combinations of the CLI arguments we execute a benchmark run. The runs of the same data set are bundled into one benchmark suite.

We first iterate through the different data sets. There we execute all the runs of the same storage formats and in these groups of runs we group them by the used consumer. The set of all runs of a data set are called a benchmark suite. The actual runs are executed by [PySysLoadBench](https://github.com/Jan-Hoc/PySysLoadBench/tree/main). Please refer to it's README to find out more regarding the structure of the benchmarking process itself and how it uses multiple processes for isolation etc. In the prerun function we make sure the data sets are converted. To account for cold runs in real world environments we do not utilize warmup rounds. For more realistic results (albeit potentially less reproducible) we leave the garbage collection active.

After all runs of a benchmark are completed, we save the raw results including system information and the used disk space, as mentioned in the metrics section. Lastly, the graphs are generated out of the results and saved too.

### Extending the Benchmark

#### Adding a Data Set

If you want to benchmark a different data set that we didn't implement, you can add your own data set file with the classes for each storage format you are interested in, as described in a previous section. Beware to think about `get_available_converters`, `get_available_pipelines` and the functions `get_tf_output_signature`, `get_tf_pad_types` and `get_collate_fn` of the `BaseDataSource`, which all must be present. The constructor of your data set classes must all atleast take the argument `path: Path | str`, specifying the location of the data set.

Additionally you need to add the data set to the dictionary `data_sources_lut` in `benchmarks/benchmarks.py`. The key you use is the one you reference in the config file with the key `dataset_type`. Also don't forget to add the data set file to the present import statement

```python
from dsi_bench.datasets import test, oadat, icon, imagenet, shapenet
```

Here you use the name of the data set file you created in `dsi_bench/datasets`. 

Lastly, so you can use the data set with the CLI argument `suites`, you need to add the value(s) you used for `suite_name` in the corresponding config file(s) to the list of the line

```python
default_suites = ['TEST', 'ICON', 'OADAT', 'ImageNet', 'ShapeNet']
```

#### Adding a Storage Format

You can also add an additional storage format to a new or existing data set. To the according file first add a data set class which is able to read from that file format and takes the same arguments in the constructor as the other classes for that data set. This should implement the same functions as well, such as `__getitem__`, `__len__` and `check_data`. It must extend the class `DataSourceBase`.

Of course you also need a conversion function to create an object of that type. It must take the arguments `source: DataSource` and `output_path: Path | str`. This should check if a conversion is already present at `output_path`, and if not convert the data set and save the conversion to `output_path` and return an instance of your new data set class.

Lastly, define a unique name for your storage format and add it as a key to the `get_available_converters` function with your conversion function as the value. Add that unique name to the variable `default_converters` in `benchmarks/benchmarks.py` so you can use it with the CLI argument `--converter_names`.

You are also free to define new DALI pipelines for your storage format, if DALI supports it. Then do not forget to add an according entry to `get_available_pipelines`.

#### Adding a Consumer

Adding an additional consumer is very similar to adding a new data set. Create a class for your consumer in `dsi_bench/consumers.py`. You are free to add helper classes for the actual consumer if it facilitates your implementation. This is done for example for `TensorflowSequenceConsumer` and `DataLoaderTensorflowSequence`.

The consumer class must implement `__call__` function, which consumes the whole data set, which is passed in the constructor. That must take the arguments `dataset: 'DataSourceBase'`, `num_threads: int`, `batch_size: int`, `device: str`. 

Lastly there are some steps in `benchmarks/benchmarks.py` to be able to use the consumer. For one, add your class to the present import statement.

```python
from dsi_bench.consumers import (
	PythonConsumer,
	TensorflowConsumer,
	PytorchConsumer,
	TensorflowSequenceConsumer,
	DALIPytorchConsumer,
	DALITensorflowConsumer,
)
```

Then, choose a unique name for your consumer and add it to `default_consumers`. Lastly, add it to `consumer_lut`, where you map the name to the actual consumer class.