from __future__ import annotations
# suppress TensorFlow warnings
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TQDM_DISABLE_AUTO"] = "true"

# Force TensorFlow to use all CPU cores for parallel ops
#os.environ['TF_NUM_INTRAOP_THREADS'] = '0'  # 0 = auto (use all cores)
#os.environ['TF_NUM_INTEROP_THREADS'] = '0'  # 0 = auto (use all cores)
#os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # OpenMP threads

import sys
sys.path.append('../')
import Preprocess2 
import Sample
import Plot_helper
from lr_find import LrFinder

import datetime
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
sys.path.pop()
sys.path.append('.././Final_solution')
import Bijectors, MAF_spline, RealNVP, Trainer, ConditionalBijectorWrapper , Utils
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any
import h5py
from pathlib import Path
from tqdm import tqdm
import math
import time
import subprocess
import matplotlib.pyplot as plt
import logging

# ==================== Suppress TensorFlow Warnings ====================
logging.getLogger('tensorflow').setLevel(logging.ERROR)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass
tf.get_logger().propagate = False

# ==================== Set up TensorFlow GPU memory management ====================
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Force eager execution to avoid XLA compilation issues with high-dimensional data
# This disables @tf.function graph compilation
tf.config.run_functions_eagerly(True)
tf.config.optimizer.set_jit(False)


mother_output_dir = '../outputs/'

# ==================== Evaluation acceleration switches ====================
# Enable the TensorFlow-backed (vectorized, multi-threaded) metrics path by converting
# numeric banks into tfp.distributions.Empirical and setting use_tf=True.
# This avoids the single-core SciPy loops used in the NumPy backend.
ENABLE_TF_PARALLEL: bool = True
TF_METRICS_DTYPE = tf.float32  # faster and lighter than float64 for metrics


# ==================== Custom TF Distribution for Bootstrap Sampling ====================
class BootstrapDistribution(tfd.Distribution):
    """
    A TensorFlow Probability distribution that wraps a fixed data bank and implements
    sampling with replacement (bootstrap). Compatible with GMetrics' generate_and_clean_data.
    
    This is necessary because tfd.Empirical causes XLA shape inference failures with
    high-dimensional data in graph mode, and we need explicit bootstrap semantics.
    """
    def __init__(self, data: Union[np.ndarray, tf.Tensor], dtype: tf.DType = tf.float32, 
                 validate_args: bool = False, allow_nan_stats: bool = True, name: str = "BootstrapDistribution"):
        """
        Args:
            data: [N, D] array of samples to bootstrap from
            dtype: output dtype for sampled tensors
        """
        parameters = dict(locals())
        
        # Convert to TF tensor and ensure correct dtype
        self._data = tf.convert_to_tensor(data, dtype=dtype)
        self._nsamples = tf.shape(self._data)[0]
        self._ndims = tf.shape(self._data)[1]
        self._dtype = dtype
        
        super(BootstrapDistribution, self).__init__(
            dtype=dtype,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name
        )
    
    def _batch_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)
    
    def _batch_shape(self):
        return tf.TensorShape([])
    
    def _event_shape_tensor(self):
        return tf.constant([self._ndims], dtype=tf.int32)
    
    def _event_shape(self):
        return tf.TensorShape([self._data.shape[1]])
    
    def _sample_n(self, n, seed=None):
        """
        Sample n points with replacement from the data bank.
        
        Args:
            n: number of samples to draw
            seed: random seed (int, scalar Tensor, or 2-element Tensor pair from GMetrics)
            
        Returns:
            samples: [n, D] tensor drawn with replacement from self._data
        """
        # GMetrics passes seed_generator.make_seeds(2)[0] which is already shape [2]
        # We just need to handle the case where someone passes a scalar for testing
        if seed is None:
            seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)
        elif isinstance(seed, int):
            seed = tf.constant([seed, 0], dtype=tf.int32)
        else:
            # Convert to tensor and ensure shape [2]
            seed = tf.convert_to_tensor(seed, dtype=tf.int32)
            seed_flat = tf.reshape(seed, [-1])
            # Pad with zeros to length 2, then slice to exactly 2
            # This works for both size=1 ([s] -> [s,0,0...][:2] = [s,0])
            # and size>=2 ([s1,s2,...] -> [s1,s2,...,0,0][:2] = [s1,s2])
            seed = tf.concat([seed_flat, tf.zeros([2], dtype=tf.int32)], axis=0)[:2]
        
        # Sample indices with replacement
        indices = tf.random.stateless_uniform(
            shape=[n],
            seed=seed,
            minval=0,
            maxval=self._nsamples,
            dtype=tf.int32
        )
        
        # Gather samples - ensure output shape is fully known at compile time
        samples = tf.gather(self._data, indices)
        # Explicitly set shape to help XLA
        D = self._data.shape[1] if self._data.shape[1] is not None else self._ndims
        samples.set_shape([None, D])
        return samples
    
    def _log_prob(self, value):
        """Not needed for bootstrap sampling, but required by Distribution interface."""
        # Uniform log-prob over data points: log(1/N) for each sample
        return tf.fill(tf.shape(value)[:-1], -tf.math.log(tf.cast(self._nsamples, self._dtype)))


def get_gpu_info() -> Optional[List[str]]:
    try:
        gpu_info: str = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]).decode('utf-8')
        return gpu_info.strip().split('\n')
    except Exception as e:
        print(e)
        return None
gpu_models: Optional[List[str]] = get_gpu_info()
if gpu_models:
    training_device: str = gpu_models[0]
    print("Successfully loaded GPU model: {}".format(training_device))
else:
    training_device = 'undetermined'
    print("Failed to load GPU model. Defaulting to 'undetermined'.")

# ==================== Load High-Level Features plotter helper ====================
sys.path.pop()
sys.path.append('/teo_fs_fast/users/acosso/Dataset/CaloChallenge_code/code')
from HighLevelFeatures import HighLevelFeatures as HLF
# Load the high-level features for the dataset
HLF_2 = HLF('electron', filename='/teo_fs_fast/users/acosso/Dataset/CaloChallenge_code/code/binning_dataset_2.xml')


def get_compiler_kwargs(lr: float,
                        ignore_nans: bool,
                        nan_threshold: float,
                        beta_1_scheduler = None,
                        clipnorm=None,
                        clipvalue = None
                       ):
    
    
    compiler_kwargs = { #'optimizer': optimizer,
                        #'optimizer': {'class_name': 'Custom>Adam',
                        'optimizer': {'class_name': 'Adam',
                                     'config': {'learning_rate': lr,
                                                'beta_1': 0.9,
                                                'beta_2': 0.999,
                                                'epsilon': 1e-07,
                                                'amsgrad': True,
                                                'clipnorm': clipnorm,
                                                'clipvalue': clipvalue}},
                       'metrics': [{'class_name': 'MinusLogProbMetric',
                                    'config': {'ignore_nans': ignore_nans,\
                                               'debug_print_mode': False}}],
                       #"compile_kwargs": {"run_eagerly": True},
                       'loss': {'class_name': 'MinusLogProbLoss',
                                'config': {'name': "MLP",
                                           'ignore_nans': ignore_nans,
                                           'nan_threshold': nan_threshold,
                                           'debug_print_mode': False}},
                       'momentum_scheduler' : {'scheduler' : beta_1_scheduler}
                        }
    return compiler_kwargs

def get_callbacks_kwargs(checkpoint_path: str,
                         es_min_delta: float,
                         es_patience: int,
                         lr_reduce_factor: float,
                         lr_min_delta: float,
                         lr_patience: int,
                         min_lr: float,
                         # ===================== new parameters =====================
                         spline_knots: int,
                         range_min: float,
                         batch_size: int,
                         x_train: tf.Tensor,
                         y_train: tf.Tensor,
                         ndims: int,
                         total_steps: int = 1000,
                         warmup_epochs: int = 18,
                         cooldown_epochs: int = 18,
                         annihilation_epochs: int = 4,
                         ):
    callbacks_kwargs = [{'class_name': 'PrintEpochInfo',
                         'config': {}},
                        #{'class_name': 'HandleNaNCallback',
                        # 'config': {'checkpoint_path': checkpoint_path,
                        #            'lr_reduction_factor': lr_reduce_factor_on_nan,
                        #            'random_seed_var': np.random.randint(1000000)}},
                        #{'class_name': 'TerminateOnNaNFractionCallback',
                        # 'config': {'threshold': 0.1,
                        #            'validation_data': X_data_val}},
                        {'class_name': 'ModelCheckpoint',
                         'config': {'filepath': checkpoint_path,
                                    'monitor': 'val_loss',
                                    'save_best_only': True,
                                    'save_weights_only': True,
                                    'verbose': 1,
                                    'mode': 'auto',
                                    'save_freq': 'epoch'}},
                        {'class_name': 'EarlyStopping',
                         'config': {'monitor': 'val_loss',
                                    'min_delta': es_min_delta,
                                    'patience': es_patience,
                                    'verbose': 1,
                                    'mode': 'auto',
                                    'baseline': None,
                                    'restore_best_weights': True}},
                        {'class_name': 'ReduceLROnPlateau',
                         'config': {'monitor': 'val_loss',
                                    'factor': lr_reduce_factor,
                                    'min_delta': lr_min_delta,
                                    'patience': lr_patience,
                                    'min_lr': min_lr}},
                        {'class_name': 'TerminateOnNaN', 'config': {}},
                        #{'class_name': 'TensorBoardWithCustomSummaries',
                        # 'config': {
                        #             'log_dir': "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                        #             'probe_y': x_train[:batch_size].numpy(),
                        #             'probe_cond': x_train[:batch_size, ndims:].numpy(),
                        #             'spline_knots': spline_knots,
                        #             'range_min': range_min,
                        #             'histogram_freq': 0,
                        #             'write_graph': False,
                        #             'update_freq': 'batch',
                        #             'profile_batch': 0}}
                        #{'class_name': 'CatchAndDebugNaN_RQS',
                        # 'config': {'spline_knots': spline_knots,
                        #            'range_min':   range_min,
                        #            'batch_size':  batch_size,
                        #            'ndims' : ndims, 
                        #            # the x_train & y_train values are passed through so the
                        #            # callback can build its own tf.data.Dataset
                        #            'x':           x_train,
                        #            'y':           y_train}},
                        {'class_name': 'OneCycleMomentum',
                                       'config': {'optimizer': None,          # placeholder; Trainer will fill
                                                  'total_steps': int(total_steps),
                                                  'pct_start':   warmup_epochs / (warmup_epochs + cooldown_epochs + annihilation_epochs),  # 18/40 = 0.45
                                                  'max_momentum': 0.95,
                                                  'base_momentum': 0.85}}
                                                         ]
    return callbacks_kwargs

def get_fit_kwargs(batch_size: int,
                   epochs_input: int,
                   validation_data: Tuple[Union[np.ndarray,tf.Tensor],Union[np.ndarray,tf.Tensor]],
                   shuffle: bool,
                   verbose: int
                  ) -> Dict[str,Any]:
    fit_kwargs = {'batch_size': batch_size,
                  'epochs': epochs_input,
                  'validation_data': validation_data,
                  'shuffle': shuffle,
                  'verbose': verbose}
    return fit_kwargs

def get_io_kwargs(path_to_results: str) -> Dict[str,Any]:
    return {'results_path': path_to_results,
            'load_weights': True,
            'load_results': True,}


layer_energies = True
MODE = 'A' 


bijector_name = 'MsplineN'
spline_knots = 8
range_min = -14
eps_regulariser = 1e-4
regulariser = 'l2'
if MODE == 'A':
    ndims = 10 
    ncond = 31 if layer_energies else 26
else: 
    ndims = 12
    ncond = 57 if layer_energies else 56
n_hidden=[128, 128]
num_bijectors = 4
activation='relu'
kernel_constraint = None
#kernel_constraint = tf.keras.constraints.MaxNorm(5.0)
bias_initializer=tf.keras.initializers.Constant(0.01)
#bias_initializer = tf.keras.initializers.Zeros()
MAF = Bijectors.ChooseCondBijector(bijector_name, ndims, spline_knots, 
                            num_bijectors, range_min, n_hidden, activation, 
                            regulariser, eps_regulariser, conditional_event_shape=(ncond,),
                            input_structure = None, 
                            bias_initializer=bias_initializer, 
                            kernel_constraint=kernel_constraint,
                            shuffle = 'RandomShuffle',
                            perm_style = 'reverse',
                            batch_norm=False,
                            conditional_input_layers='first_layer',
                            tails='linear')



base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(ndims, dtype=tf.float64), scale_diag=tf.ones(ndims, dtype=tf.float64))


# # Trainer init, will only function as a data loader
if MODE == 'A':
    D_total = 41 if layer_energies else 36 
else:
    D_total = 69 if layer_energies else 68 
precision = tf.float64 

dummy_sample = tf.zeros([1, D_total], dtype=precision)



# ## Init


run_number: int = 53
n_runs: int = 1

### Base run directory ###
path_to_results , _ = Utils.define_run_dir(mother_output_dir+'run_'+str(run_number)+'/',
                                           force = "skip",
                                           bkp = False)

### Compiler hyperparameters ###
lr: float = 5e-5
ignore_nans: bool = False
nan_threshold: float = 0.01

### Initialize callbacks hyperparameters ###
path_to_weights: str = Utils.define_dir(os.path.join(path_to_results, 'weights'))
checkpoint_path: str = os.path.join(path_to_weights, 'best_weights.h5')
es_min_delta: float = .0001
es_patience: int = 100
lr_reduce_factor: float = .5
lr_min_delta: float = .0001
lr_patience: int = 50
min_lr: float = 1e-6

### Initialize training hyperparameters ###
batch_size: int =  60000
epochs_input: int = 40
shuffle: bool = True
verbose_trainer: int = 2

### Debugging parameter
debug_print_mode: bool = False


from importlib import reload
#reload(Trainer)

NFObject: Trainer.Trainer = Trainer.Trainer(base_distribution = base_dist,
                                        flow = MAF, 
                                        x_data_train = dummy_sample,
                                        y_data_train = dummy_sample,
                                        io_kwargs = get_io_kwargs(path_to_results = path_to_results),
                                        compiler_kwargs = get_compiler_kwargs(lr = lr,
                                                                                ignore_nans = True,
                                                                                nan_threshold = nan_threshold),
                                        callbacks_kwargs = get_callbacks_kwargs(checkpoint_path = checkpoint_path,
                                                                                es_min_delta = es_min_delta,
                                                                                es_patience = es_patience,
                                                                                lr_reduce_factor = lr_reduce_factor,
                                                                                lr_min_delta = lr_min_delta,
                                                                                lr_patience = lr_patience,
                                                                                min_lr = min_lr,
                                                                                batch_size = batch_size,
                                                                                spline_knots = spline_knots,
                                                                                range_min = range_min,
                                                                                x_train = dummy_sample,
                                                                                y_train = dummy_sample,
                                                                                ndims = ndims,),
                                        fit_kwargs = get_fit_kwargs(batch_size = batch_size,
                                                                    epochs_input = epochs_input,
                                                                    validation_data = (dummy_sample, dummy_sample),
                                                                    shuffle = shuffle,
                                                                    verbose = verbose_trainer),
                                        debug_print_mode = debug_print_mode)
trainable_params: int = NFObject.trainable_params
non_trainable_params: int = NFObject.non_trainable_params


print("building and caching evaluation dataset...")
MODE = 'A'
showers_raw, cs, E_inc, fine_flat, cond_flat, factors = Preprocess2.build_and_cache(
    raw_h5     = "/teo_fs_fast/users/acosso/Dataset/dataset_2_2.hdf5",
    cache_file = "/teo_fs_fast/users/acosso/Dataset/eval_dataset_A.hdf5",
    mode       = MODE,
    E_coarse_max_ref = None,  # Set to None to use the maximum from the dataset
    force_rebuild = False,
    verbose = True, 
    noise_scale=1e-4,
    compression = 'lzf'
)
print("done.")

sys.path.append('../')
import GMetrics # type: ignore
from GMetrics.plotters import plot_corners, cornerplotter, plot_corr_matrix, plot_corr_matrix_side_by_side # type: ignore
from GMetrics.more import deformations # type: ignore
from GMetrics.more.optimizers_deformations import compute_exclusion_bisection, compute_exclusion_LR_bisection # type: ignore
from GMetrics.utils import se_mean, se_std, NumpyDistribution
from GMetrics.notebooks import shared
try:
    from GMetrics.base import NumpyDistribution
except ImportError:
    try:
        from GMetrics.more import NumpyDistribution
    except ImportError:
        from GMetrics.utils import NumpyDistribution
import pandas as pd
from eval_streaming import evaluate_flow_and_cache
from make_eval_features import build_eval_features

print("evaluating and caching flow results...")
evaluate_flow_and_cache(
    in_cache_h5="/teo_fs_fast/users/acosso/Dataset/eval_dataset_A.hdf5",
    out_h5="/teo_fs_fast/users/acosso/Dataset/eval_results_A.hdf5",
    flow=MAF,
    base_dist=base_dist,
    gpu_batch=65536,          # lower if you still see OOM (e.g. 16384)
    io_chunk_rows=2_000_000,  # disk I/O window; 0.5–2M is usually sweet-spot
    start_row=0,
    n_rows=None,              # None = process to the end
    tf_dtype=tf.float32,
    save_as_float32=True,     # shrinks result file by ~2x
    clip_latent=False,
    latent_clip_value=4.0,
    compression="lzf",
    seed=42,
    overwrite=False,
)
print("done.")

print("building and caching evaluation features...")
meta = build_eval_features(
    results_h5="/teo_fs_fast/users/acosso/Dataset/eval_results_A.hdf5",
    truth_h5="/teo_fs_fast/users/acosso/Dataset/eval_dataset_A.hdf5",
    out_h5="/teo_fs_fast/users/acosso/Dataset/eval_features_A.hdf5",
    io_chunk_events=100_000,
    compression="lzf",
    overwrite=False,
)
print("done.")




from GMetrics.utils import NumpyDistribution
class EmpiricalNumpyDistribution(NumpyDistribution):
    def __init__(self, data: np.ndarray, dtype: np.dtype = np.float32):
        """
        Initialize with your numeric data and ensure the dtype is consistent.
        
        Args:
            data (np.ndarray): The empirical data to sample from.
            dtype (np.dtype): The desired data type for the output samples.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy ndarray.")
        
        self.data = data
        self.dtype = dtype
        self.ndims = data.shape[1]  # Number of features (columns)
        self.nsamples = data.shape[0]  # Number of samples (rows)
    
    def sample(self, n: int, seed: int = None) -> np.ndarray:
        """
        Sample `n` data points with replacement from the stored data.

        Args:
            n (int): The number of samples to return.
            seed (int): The seed for random number generation to ensure reproducibility.

        Returns:
            np.ndarray: A batch of resampled data with shape (n, self.ndims).
        """
        # Use the seed to make the sampling deterministic
        rng = np.random.default_rng(seed)

        # Resample with replacement
        indices = rng.choice(self.nsamples, size=n, replace=True)
        resampled_data = self.data[indices]
        
        # Ensure the output has the desired dtype
        return resampled_data.astype(self.dtype)


# # Evaluating full showers: 6480 dimensions
# ## directory path

#model_dir = "./53/"
model_dir = os.path.join("./", str(run_number) + "/Full_showers_eval_tf_batchsize_10k_nslices_1k/")
if os.path.exists(model_dir):
    print("Model directory exists")
else:
    os.makedirs(model_dir)
    print("Model directory was created")
null_hypotheses_dir = model_dir + "null_hypothesis/"
if os.path.exists(null_hypotheses_dir):
    print("Null hypothesis directory exists")
else:
    os.makedirs(null_hypotheses_dir)
    print("Null hypothesis directory was created")
metrics_config_file = model_dir + "metrics_config.json"


# ## loading data


# === load_full_showers_from_segments.py ===
from pathlib import Path

def load_full_showers(
    results_h5: str,
    *,
    n_events: int = 100_000,
    ncoarse_per_event: Optional[int] = None,  # if None, auto-detect from file attrs (V)
    dtype: str | np.dtype = "float32",
    events_chunk: int = 1024,              # assemble this many events per streaming chunk
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble full calorimeter showers from row-wise (coarse, fine-slice) segments.

        Expects datasets:
            /truth_E : [N_rows, Df_per_coarse]   where N_rows = total_events * ncoarse_per_event
            /model_E : [N_rows, Df_per_coarse]

        For Mode A, Df_per_coarse = 10 and ncoarse_per_event = 648 → full shower has 6480 voxels.

    Returns
    -------
    input_true : (n_events, ncoarse_per_event * Df_per_coarse) float32
    input_gen  : (n_events, ncoarse_per_event * Df_per_coarse) float32
    """
    path = Path(results_h5)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with h5py.File(path, "r") as f:
        if "/truth_E" not in f or "/model_E" not in f:
            raise KeyError("Expected '/truth_E' and '/model_E' in results file.")

        ds_true = f["/truth_E"]   # [N_rows, Df_per_coarse]
        ds_gen  = f["/model_E"]   # [N_rows, Df_per_coarse]
        N_rows, Df_per = ds_true.shape
        if ds_gen.shape != (N_rows, Df_per):
            raise ValueError("Shape mismatch between '/truth_E' and '/model_E'.")

        # --- Robust auto-detection of ncoarse_per_event ---
        if ncoarse_per_event is None:
            attr_V = f.attrs.get("V", None)  # should be nz*na*nr = 648 for Mode A
            candidate_from_rows = None
            if N_rows % n_events == 0:  # user requested n_events (default 100k)
                candidate_from_rows = N_rows // n_events
            # Build list of plausible candidates
            candidates = []
            if attr_V is not None:
                candidates.append(int(attr_V))
            if candidate_from_rows is not None:
                candidates.append(int(candidate_from_rows))
            # Common mistaken stored value: attr_V * Df_per (e.g. 648*10=6480)
            if attr_V is not None and attr_V % Df_per == 0:
                wrong_mult = int(attr_V * Df_per)
                if N_rows % wrong_mult == 0:  # treat as potential mis-specified
                    candidates.append(int(attr_V))  # keep correct one
            # If attribute absent, try to infer by factoring N_rows.
            if not candidates:
                # Heuristic: try typical geometry 648 first
                if N_rows % 648 == 0:
                    candidates.append(648)
                # fallback: scan small divisors up to 2000 and pick one near 648 if exists
                for d in (576, 600, 640, 648, 672, 720):
                    if N_rows % d == 0:
                        candidates.append(d)
                if not candidates:
                    raise ValueError(
                        f"Cannot infer ncoarse_per_event: no attrs 'V', and no typical divisor fits N_rows={N_rows}."
                    )
            # Choose candidate closest to 648 (expected) if multiple
            ncoarse_per_event = min(candidates, key=lambda c: abs(c - 648))
            if verbose:
                print(f"[load_full_showers] Inferred ncoarse_per_event={ncoarse_per_event} (candidates={candidates}).")
        # Detect common error: using 10x the correct value (6480)
        if ncoarse_per_event >= 5000 and ncoarse_per_event % 10 == 0:
            suspected = ncoarse_per_event // 10
            if N_rows % suspected == 0 and suspected in (648, 576, 640, 672, 720):
                if verbose:
                    print(f"[load_full_showers] Warning: ncoarse_per_event={ncoarse_per_event} looks like 10x too large; correcting to {suspected}.")
                ncoarse_per_event = suspected

        # Total events available in the file (must be an integer multiple of ncoarse_per_event).
        if N_rows % ncoarse_per_event != 0:
            raise ValueError(
                f"N_rows={N_rows} not divisible by inferred ncoarse_per_event={ncoarse_per_event}. Df_per={Df_per}."
            )
        total_events = N_rows // ncoarse_per_event
        if verbose:
            print(f"[load_full_showers] total_events={total_events}, full_dimension={ncoarse_per_event*Df_per}")

        # Cap n_events to what's available
        n = min(n_events, total_events)
        if verbose:
            print(
                f"Assembling {n} / {total_events} events "
                f"(rows={N_rows}, ncoarse={ncoarse_per_event}, Df_per={Df_per} → full D={ncoarse_per_event*Df_per}) "
                f"from {path.name}"
            )

        full_D = ncoarse_per_event * Df_per
        out_true = np.empty((n, full_D), dtype=dtype)
        out_gen  = np.empty((n, full_D), dtype=dtype)

        # Stream in blocks of whole events
        ev = 0
        while ev < n:
            k = min(events_chunk, n - ev)  # events in this chunk
            # Corresponding row slice (each event contributes ncoarse_per_event rows)
            i0 = ev * ncoarse_per_event
            i1 = (ev + k) * ncoarse_per_event

            # Read contiguous rows for these k events
            block_true = ds_true[i0:i1, :]   # shape = (k * ncoarse_per_event, Df_per)
            block_gen  = ds_gen[i0:i1, :]

            # Reshape to (k, ncoarse, Df_per) then flatten to (k, ncoarse*Df_per)
            block_true = block_true.reshape(k, ncoarse_per_event, Df_per).astype(dtype, copy=False)
            block_gen  = block_gen.reshape(k, ncoarse_per_event, Df_per).astype(dtype, copy=False)

            out_true[ev:ev+k, :] = block_true.reshape(k, full_D)
            out_gen[ev:ev+k, :]  = block_gen.reshape(k, full_D)

            ev += k
            if verbose:
                print(f"  -> assembled {ev}/{n} events")

    return out_true, out_gen


print("loading full showers from segments...")
input_true, input_gen = load_full_showers(
    results_h5="/teo_fs_fast/users/acosso/Dataset/eval_results_A.hdf5",
    n_events=100_000,             # target events
    # ncoarse_per_event left as None → auto-detected from file attrs (V)
    dtype="float32",
    events_chunk=1024,            # tune if you want larger or smaller memory bursts
)
print("shape of the input: ",input_true.shape, input_gen.shape, input_true.dtype)
# (100000, 6480) (100000, 6480) float32
print("done.")


orig_true = input_true.copy()
rng1 = np.random.default_rng(1234)
rng2 = np.random.default_rng(5678)
shuffled_true1 = orig_true[rng1.permutation(orig_true.shape[0])]
shuffled_true2 = orig_true[rng2.permutation(orig_true.shape[0])]
input_full_showers_true1 = EmpiricalNumpyDistribution(shuffled_true1.copy(), dtype=np.float32)
input_full_showers_true2 = EmpiricalNumpyDistribution(shuffled_true2.copy(), dtype=np.float32)



# ## input builder 

seed_test = 42
batch_size_test = 10000
niter = 1000

# Use BootstrapDistribution to enable bootstrap resampling - this allows unlimited
# sampling from the fixed data bank, avoiding GMetrics' batch_size reduction
dist_true_bootstrap = BootstrapDistribution(orig_true, dtype=TF_METRICS_DTYPE)
TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(
    dist_1_input = dist_true_bootstrap,  # TFP distribution with bootstrap sampling
    dist_2_input = dist_true_bootstrap,  # same for null (truth vs truth)
    niter = niter,
    batch_size_test = batch_size_test,
    batch_size_gen = 1,
    small_sample_threshold = 1,
    dtype_input = TF_METRICS_DTYPE,
    seed_input = seed_test,
    use_tf = True,
    mirror_strategy = False,
    verbose = True,
)
print("nsamples",TwoSampleTestInputs.nsamples)
print("batch_size",TwoSampleTestInputs.batch_size_test)
print("niter",TwoSampleTestInputs.niter)
print("niter * batch_size",TwoSampleTestInputs.niter*TwoSampleTestInputs.batch_size_test)
print("small_sample",TwoSampleTestInputs.small_sample)


# ## metric init


nslices = 1000
KSTest = GMetrics.KSTest(TwoSampleTestInputs,
                         progress_bar = True,
                         verbose = True)
SKSTest = GMetrics.SKSTest(TwoSampleTestInputs,
                           nslices = nslices, # to be included in metric kwargs
                           seed_slicing = 0, # to be included in metric kwargs
                           progress_bar = True,
                           verbose = True)
SWDMetric = GMetrics.SWDMetric(TwoSampleTestInputs,
                               nslices = nslices, # to be included in metric kwargs
                               seed_slicing = 0, # to be included in metric kwargs
                               progress_bar = True,
                               verbose = True)

test_config_null = {}
test_config_tmp = dict(TwoSampleTestInputs.__dict__)
keys_to_remove = ["_dist_1_input", "_dist_2_input", "_dist_1_num", "_dist_2_num", "_dist_1_symb", "_dist_2_symb", "_seed_generator"]
for key in keys_to_remove:
    test_config_tmp.pop(key, None)
for key, value in test_config_tmp.items():
    new_key = key.lstrip('_')
    
    if isinstance(value, tf.Tensor):
        new_value = value.numpy() # type: ignore
    elif isinstance(value, np.ndarray):
        new_value = value.tolist()
    elif isinstance(value, np.generic):
        new_value = value.item() # Convert NumPy scalars to Python scalars
    elif isinstance(value, tf.DType):
        new_value = value.name
    elif isinstance(value, np.dtype):
        new_value = np.dtype(value).name
    else:
        new_value = value
    
    test_config_null[new_key] = new_value


ndims = TwoSampleTestInputs.ndims
unique_key = "config_ndims_"+str(TwoSampleTestInputs.ndims)+"_nsamples_"+str(TwoSampleTestInputs.batch_size_test)+"_niter_"+str(TwoSampleTestInputs.niter)
metrics_config = {unique_key: {
                               "ks":  {"name": "ks",
                                       "object_name": "KSTest",
                                       "class_name": "GMetrics.KSTest", 
                                       "kwargs": {},
                                       "result_key": "statistic_means", 
                                       "scale_func": lambda ns, _ : np.sqrt(ns),
                                       "scale_func_string": "lambda ns, _: np.sqrt(ns)",
                                       "test_config": test_config_null,
                                       "max_vectorize": 10000,
                                       "latex": "$t_{\overline{\mathrm{KS}}}$",
                                       "null_file": null_hypotheses_dir+"KS.json"},
                               "sks": {"name": "sks",
                                       "object_name": "SKSTest",
                                       "class_name": "GMetrics.SKSTest", 
                                       "kwargs": {"nslices": nslices, 
                                                  "seed_slicing": 0},
                                       "result_key": "metric_means", 
                                       "scale_func": lambda ns, _ : np.sqrt(ns),
                                       "scale_func_string": "lambda ns, _: np.sqrt(ns)",
                                       "test_config": test_config_null,
                                       "max_vectorize": 10,
                                       "latex": "$t_{\mathrm{SKS}}$",
                                       "null_file": null_hypotheses_dir+"SKS.json"},
                               "swd": {"name": "swd",
                                       "object_name": "SWDMetric",
                                       "class_name": "GMetrics.SWDMetric", 
                                       "kwargs": {"nslices": nslices, 
                                                  "seed_slicing": 0},
                                       "result_key": "metric_means", 
                                       "scale_func": lambda ns, ndims: np.sqrt(ns/ndims),
                                       "scale_func_string": "lambda ns, ndims: np.sqrt(ns/ndims)",
                                       "test_config": test_config_null,
                                       "max_vectorize": 10,
                                       "latex": "$t_{\mathrm{SW}}$",
                                       "null_file": null_hypotheses_dir+"SWD.json"}}}


# ## Null hypothesis evaluation

print("evaluating null hypotheses...")
import inspect

for metric in list(metrics_config[unique_key].values()):
    file = metric["null_file"]
    name = metric["name"]
    max_vectorize = metric.get("max_vectorize", None)
    obj = eval(metric["object_name"])

    os.makedirs(os.path.dirname(file), exist_ok=True)

    use_tf_flag = getattr(obj.Inputs, "use_tf", False)
    test_fn = getattr(obj, "Test_tf", None) if use_tf_flag else getattr(obj, "Test_np", None)
    if test_fn is None:
        raise RuntimeError(f"{name}: no test function for use_tf={use_tf_flag}")

    if os.path.exists(file):
        print(f"Loading {name} from {file}")
        obj.Results.load_from_json(file)
    else:
        print(f"Computing and saving {name} (use_tf={use_tf_flag})")
        sig = inspect.signature(test_fn)
        call_kwargs = {}
        if "max_vectorize" in sig.parameters and max_vectorize is not None:
            call_kwargs["max_vectorize"] = max_vectorize
        test_fn(**call_kwargs)
        print(f"Saving {name} to {file}")
        obj.Results.save_to_json(file)
print("done.")

cl_list = [0.68, 0.95, 0.99]
null_times = []
print("computing thresholds...")
for metric in list(metrics_config[unique_key].values()):
    name = metric["name"]
    obj = eval(metric["object_name"])
    result_key = metric["result_key"]
    scale_func = metric["scale_func"]
    
    #params = obj.Results[-1].__dict__  # parameters are in the result object
    #bs_used = int(params.get('batch_size_test_used', obj.Inputs.batch_size_test))
    #ns_eff = bs_used / 2.0
    #dist_null = np.array(obj.Results[-1].result_value[result_key]) * scale_func(ns_eff, ndims)
    nsamples = obj.Inputs.batch_size_test
    ns = nsamples**2 /(2*nsamples)  # == nsamples/2
    dist_null = np.array(obj.Results[-1].result_value[result_key]) * scale_func(ns, ndims)

    metric_thresholds = [[cl, 
                          [int(cl*len(dist_null)), 
                           int((1-cl)*len(dist_null))], 
                          np.sort(dist_null)[int(len(dist_null)*cl)]] for cl in cl_list]
    print(f"ThresholdS for metric {metric['name']}: {metric_thresholds}")
    null_time = obj.Results[-1].__dict__['computing_time']
    null_times.append([name, int(null_time)])
    print(f"Computing time for metric {metric['name']}: {null_time}")
    
    metric.update({"thresholds": metric_thresholds})
GMetrics.utils.save_update_metrics_config(metrics_config = metrics_config, metrics_config_file = metrics_config_file) # type: ignore
print("done.")

print("computing observed distributions...")
def compute_observed_distributions(
    metrics_config: dict,
    unique_key: str,
    dist_1_num,                 # TRUTH bank [Ns, D], np.ndarray or tf.Tensor (numeric) OR NumpyDistribution
    dist_2_num,                 # MODEL  bank [Ns, D], np.ndarray or tf.Tensor (numeric) OR NumpyDistribution
    ndims: int,
    batch_size_test: int,
    niter_obs: int = 100,
    dtype=tf.float32,
    use_tf: bool = ENABLE_TF_PARALLEL,  # default to TF path if enabled globally
    verbose: bool = False,
    *,
    cache_file: str | None = None,  # optional: load/save a pickle cache
    overwrite: bool = False,
    show_progress: bool = True,     # external tqdm over metrics
):
    """
    Builds observed (alternative) distributions T_obs under H1 by repeatedly
    sampling truth vs model with n=m=batch_size_test, niter_obs times.

    Now supports NumPy-side symbolic distributions (e.g. EmpiricalNumpyDistribution),
    in addition to numeric banks. Logic is unchanged for numeric inputs.

    Caching:
      - If `cache_file` is provided and exists (and overwrite=False), the cached
        results are loaded and immediately returned (no computation).
      - Otherwise, results are computed as usual and then saved to `cache_file`
        (if provided).

    Returns:
      results[name] = {
         'alt_array': np.ndarray shape [niter_obs],   # scaled statistics
         'alt_mean': float,
         'alt_std':  float
      }
    """
    import os, pickle, time, inspect
    import numpy as np

    # --- helpers: detect numeric arrays/tensors vs NumPy distributions ---
    def _is_numeric_bank(x):
        try:
            import tensorflow as _tf
            return isinstance(x, (np.ndarray, _tf.Tensor))
        except Exception:
            return isinstance(x, np.ndarray)

    def _is_numpy_distribution(x):
        # Prefer a strong isinstance check if NumpyDistribution is importable;
        # otherwise fall back to a light duck-typing check.
        try:
            from Thesis.evaluation.utils_eval import NumpyDistribution as _ND  # same module that defines EmpiricalNumpyDistribution's base
            return isinstance(x, _ND)
        except Exception:
            return (hasattr(x, "sample") and callable(getattr(x, "sample"))
                    and not hasattr(x, "shape"))  # distributions don't expose .shape like arrays/tensors

    # ---------- Fast path: load from cache if available ----------
    if cache_file and (not overwrite) and os.path.isfile(cache_file):
        if verbose or show_progress:
            print(f"[compute_observed_distributions] Loaded cached results from: {cache_file}")
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        return payload.get("results", {})

    # ---------- Sanity checks (only for fully numeric banks) ----------
    both_numeric = _is_numeric_bank(dist_1_num) and _is_numeric_bank(dist_2_num)
    if both_numeric:
        N1, N2 = int(dist_1_num.shape[0]), int(dist_2_num.shape[0])
        assert N1 >= batch_size_test and N2 >= batch_size_test, (
            f"batch_size_test={batch_size_test} exceeds available rows: truth={N1}, model={N2}."
        )
        if niter_obs * batch_size_test > min(N1, N2) and verbose:
            print("[compute_observed_distributions] Note: iterations will reuse rows across batches "
                  f"(niter×batch={niter_obs*batch_size_test} > min(Ns)={min(N1,N2)}).")
    else:
        # If one or both inputs are NumPy distributions, we cannot pre-check sizes.
        # Sampling feasibility and any adjustments are handled internally by GMetrics.
        if verbose:
            kinds = (
                "dist_1: " + ("numeric" if _is_numeric_bank(dist_1_num) else
                              "numpy-distribution" if _is_numpy_distribution(dist_1_num) else "unknown"),
                "dist_2: " + ("numeric" if _is_numeric_bank(dist_2_num) else
                              "numpy-distribution" if _is_numpy_distribution(dist_2_num) else "unknown"),
            )
            print("[compute_observed_distributions] Using generalized inputs ->", ", ".join(kinds))

    # ---------- Build observed inputs ----------
    # Wrap in BootstrapDistribution to enable bootstrap resampling for TF path
    def _to_bootstrap_dist(x):
        """Convert NumpyDistribution or array to BootstrapDistribution for TF backend."""
        if isinstance(x, np.ndarray):
            return BootstrapDistribution(x, dtype=dtype)
        if hasattr(x, "data") and isinstance(x.data, np.ndarray):
            return BootstrapDistribution(x.data, dtype=dtype)  # EmpiricalNumpyDistribution.data
        return x  # fallback for already-TF distributions

    dist1_bootstrap = _to_bootstrap_dist(dist_1_num)
    dist2_bootstrap = _to_bootstrap_dist(dist_2_num)
    ObsInputs = GMetrics.TwoSampleTestInputs(
        dist_1_input    = dist1_bootstrap,  # BootstrapDistribution
        dist_2_input    = dist2_bootstrap,  # BootstrapDistribution
        niter           = niter_obs,
        batch_size_test = batch_size_test,
        small_sample_threshold = 1,
        dtype_input     = dtype,
        seed_input      = 43,
        use_tf          = True,
        mirror_strategy = False,
        verbose         = verbose,
    )

    # ---------- Progress iterator over metrics ----------
    try:
        from tqdm.auto import tqdm
        metric_items = list(metrics_config[unique_key].items())
        iter_metrics = tqdm(metric_items, disable=not show_progress,
                            desc=f"T_obs (niter={niter_obs}, batch={batch_size_test})")
    except Exception:
        iter_metrics = list(metrics_config[unique_key].items())

    results = {}

    # ---------- Loop metrics ----------
    for name, meta in iter_metrics:
        lname = name.lower()
        if lname in {"lr", "likelihood_ratio", "likelihoodratio"}:
            if show_progress and hasattr(iter_metrics, "write"):
                iter_metrics.write(f("[skip] {name} requires likelihoods; skipping in observed pass."))
            continue

        # Instantiate metric CLASS (keep your original logic)
        MetricClass   = eval(meta["class_name"])         # e.g. GMetrics.SKSTest
        metric_kwargs = dict(meta.get("kwargs", {}))
        obj = MetricClass(
            ObsInputs,
            progress_bar = True,      # let GMetrics show its own bar if available
            verbose      = verbose,
            **metric_kwargs
        )

        # Choose backend
        use_tf_flag = getattr(obj.Inputs, "use_tf", False)
        test_fn = getattr(obj, "Test_tf", None) if use_tf_flag else getattr(obj, "Test_np", None)
        if test_fn is None:
            raise RuntimeError(f"{name}: no test function for use_tf={use_tf_flag}")

        # Optional max_vectorize
        call_kwargs = {}
        import inspect as _inspect
        sig = _inspect.signature(test_fn)
        max_vectorize = meta.get("max_vectorize", None)
        if ("max_vectorize" in sig.parameters) and (max_vectorize is not None):
            call_kwargs["max_vectorize"] = max_vectorize

        t0 = time.time()
        if show_progress and hasattr(iter_metrics, "set_postfix_str"):
            iter_metrics.set_postfix_str(f"running {name}…")

        # Run once; GMetrics fills obj.Results[-1] with niter_obs replicates
        test_fn(**call_kwargs)

        t1 = time.time()

        # Extract replicates and apply SAME scaling as for nulls
        result_key  = meta["result_key"]   # e.g. 'metric_means', 'statistic_means', 'metric_list'
        scale_func  = meta.get("scale_func", None)
        raw_arr     = np.array(obj.Results[-1].result_value[result_key])

        # Effective ns: keep your original convention (n/2 for two-sample tests)
        ns_eff      = batch_size_test / 2.0
        scale       = scale_func(ns_eff, ndims) if callable(scale_func) else 1.0
        arr_scaled  = raw_arr * scale

        if lname == "fgd":
            arr_scaled = arr_scaled * 10.0  # keep your plotting convention

        results[name] = {
            "alt_array": arr_scaled,
            "alt_mean":  float(arr_scaled.mean()),
            "alt_std":   float(arr_scaled.std(ddof=1)) if arr_scaled.size > 1 else 0.0,
        }

        if show_progress and hasattr(iter_metrics, "write"):
            iter_metrics.write(f"[done] {name} in {t1 - t0:.2f}s  (niter={niter_obs}, batch={batch_size_test})")

    # ---------- Save cache (optional) ----------
    if cache_file:
        payload = {
            "results": results,
            "meta": {
                "unique_key": unique_key,
                "ndims": ndims,
                "batch_size_test": batch_size_test,
                "niter_obs": niter_obs,
                "dtype": str(dtype),
                "use_tf": use_tf,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": list(metrics_config[unique_key].keys()),
                # Note: inputs could be numeric or symbolic; we don't serialize them here.
            },
        }
        try:
            d = os.path.dirname(cache_file)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose or show_progress:
                print(f"[compute_observed_distributions] Cached results saved to: {cache_file}")
        except Exception as e:
            if verbose or show_progress:
                print(f"[compute_observed_distributions] WARNING: failed to save cache: {e}")

    return results


orig_true = input_true.copy()
orig_gen = input_gen.copy()
rng1 = np.random.default_rng(1)
rng2 = np.random.default_rng(2)
shuffled_true = orig_true[rng1.permutation(orig_true.shape[0])]
shuffled_gen = orig_gen[rng2.permutation(orig_gen.shape[0])]
input_full_showers_true = EmpiricalNumpyDistribution(shuffled_true.copy(), dtype=np.float32)
input_full_showers_gen = EmpiricalNumpyDistribution(shuffled_gen.copy(), dtype=np.float32)

# input data to be loaded
cache = model_dir + "/alternative_hypothesis/atl_hyp_full_10k_1k"

observed = compute_observed_distributions(
    metrics_config, unique_key,
    dist_1_num=input_full_showers_true,
    dist_2_num=input_full_showers_gen,
    ndims=ndims,
    batch_size_test=batch_size_test,
    niter_obs=1000,
    dtype=TF_METRICS_DTYPE, use_tf=ENABLE_TF_PARALLEL, verbose=False,
    cache_file=cache, overwrite=True, show_progress=True
)
print("done.")

