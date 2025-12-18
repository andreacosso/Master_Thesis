# suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TQDM_DISABLE_AUTO"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("GPU_ID", "0")  # default to GPU #0
import sys
import Thesis.deprec_Preprocess2 as deprec_Preprocess2 
import Sample
import Plot_helper
import datetime
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

sys.path.append('/auto_home/users/acosso/Thesis/Final_solution')
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


# ==================== Load High-Level Features plotter helper ====================
sys.path.pop()
sys.path.append('/teo_fs_fast/users/acosso/Dataset/CaloChallenge_code/code')
from HighLevelFeatures import HighLevelFeatures as HLF
# Load the high-level features for the dataset
HLF_2 = HLF('electron', filename='/teo_fs_fast/users/acosso/Dataset/CaloChallenge_code/code/binning_dataset_2.xml')

# ==================== Define the mother output directory ====================
mother_output_dir = '/auto_home/users/acosso/Thesis/outputs/'

# ==================== Get GPU information ====================
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



def main():
    # # Trainer helper functions
    # 

    # ## Complire kwargs
    def get_compiler_kwargs(lr: float,
                            ignore_nans: bool,
                            nan_threshold: float,
                            beta_1_scheduler = None,
                            clipnorm=None,
                            clipvalue = None
                        ):


        compiler_kwargs = { #'optimizer': optimizer,
                            'optimizer': {'class_name': 'Adam', # this gives the new Adam optimizer
                                        'config': {'learning_rate': lr,
                                                    'beta_1': 0.9,
                                                    'beta_2': 0.999,
                                                    'epsilon': 1e-08,
                                                    'amsgrad': False,
                                                    'clipnorm': clipnorm,
                                                    'clipvalue': clipvalue}},
                        'metrics': [{'class_name': 'MinusLogProbMetric',
                                        'config': {'ignore_nans': ignore_nans,
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


    # ## Callback kwargs
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
                             total_steps: int,
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
                            #{'class_name': 'ReduceLROnPlateau',
                            # 'config': {'monitor': 'val_loss',
                            #            'factor': lr_reduce_factor,
                            #            'min_delta': lr_min_delta,
                            #            'patience': lr_patience,
                            #            'min_lr': min_lr}},
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
                            #{'class_name': 'OneCycleMomentum',
                            # 'config': {'optimizer': None,          # placeholder; Trainer will fill
                            #            'total_steps': int(total_steps),
                            #            'pct_start':   warmup_epochs / (warmup_epochs + cooldown_epochs + annihilation_epochs),  # 18/40 = 0.45
                            #            'max_momentum': 0.95,
                            #            'base_momentum': 0.85}}
                            ]
        return callbacks_kwargs


    # ## Fit kwargs
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


    # ## IO kwargs
    def get_io_kwargs(path_to_results: str) -> Dict[str,Any]:
        return {'results_path': path_to_results,
                'load_weights': False,
                'load_results': False,}


    # # Build the preprocessed data
    MODE = 'A'
    showers_raw, cs, E_inc, fine_flat, cond_flat, factors = deprec_Preprocess2.build_and_cache(
        raw_h5     = "/teo_fs_fast/users/acosso/Dataset/dataset_2_1.hdf5",
        cache_file = "/teo_fs_fast/users/acosso/Dataset/preproc_cache_test_A_10000.hdf5",
        mode       = MODE,
        E_coarse_max_ref = None,  # Set to None to use the maximum from the dataset
        force_rebuild = False,
        verbose = True, 
        max_events = 10_000,
        compression = 'lzf'
    )


    # ## Load the data to Train and Validation
    batch_size_loader = 2_000 
    x_train_ds, y_train_ds, x_val_ds, y_val_ds = deprec_Preprocess2.build_train_val_datasets(
            cache_file="/teo_fs_fast/users/acosso/Dataset/preproc_cache_test_A_10000.hdf5",
            batch_size=batch_size_loader,
            precision=tf.float64,
            seed=42)


    # # Bijector init
    bijector_name = 'MsplineN'
    spline_knots = 8
    range_min = -14
    eps_regulariser = 1e-3
    #regulariser = 'l2'
    regulariser = None
    if MODE == 'A':
        ndims = 10
        ncond = 31
    else: 
        ndims = 12
        ncond = 57
    n_hidden=[128, 128]
    #num_bijectors = 8
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
                                shuffle = 'RandomShuffle',
                                perm_style = 'reverse',
                                bias_initializer=bias_initializer, 
                                kernel_constraint=kernel_constraint,
                                batch_norm=False,
                                tails='linear')


    # # TODO
    # 
    # ## Now working: 
    # - togliere il clipping dove non necessario 
    # - Preprocess il dataset completo 
    # - capire se allenare solo in mode = A o anche mode = B
    # - capire come fare il preprocess del dataset di evaluation 
    # - niente inverte il **preprocess dopo il sampling** in questo momento? 
    # - aggiustare le funzioni di plot deglishowers
    # - **AGGIUNGERE TUTTA LA PARTE DI METRICHE**
    # 
    # check in case it breaks
    # - float64 **DONE**
    # - togli conditionals
    # - batch norm
    # - clipping 
    # - histo pesi 
    # - togli lr_schedule 


    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(ndims, dtype=tf.float64), scale_diag=tf.ones(ndims, dtype=tf.float64))


    # # Trainer init
    run_number: int = 39
    n_runs: int = 1

    ### Base run directory ###
    path_to_results , _ = Utils.define_run_dir(mother_output_dir+'run_'+str(run_number)+'/',
                                            force = "skip",
                                            bkp = False)

    ### Compiler hyperparameters ###
    lr: float = 5e-4
    ignore_nans: bool = False
    nan_threshold: float = 0.01

    ### Initialize callbacks hyperparameters ###
    path_to_weights: str = Utils.define_dir(os.path.join(path_to_results, 'weights'))
    checkpoint_path: str = os.path.join(path_to_weights, 'best_weights.h5')
    es_min_delta: float = .0001
    es_patience: int = 91
    lr_reduce_factor: float = .7
    lr_min_delta: float = .00001
    lr_patience: int = 10
    min_lr: float = 7e-7

    ### Initialzie training hyperparameters ###
    num_gpus = 4
    batch_size: int =  2_000
    if batch_size != batch_size_loader:
        batch_size = batch_size_loader
        print(f"Warning: batch_size {batch_size} does not match loader size {batch_size_loader}.")
        print("using batch_size_loader instead.")
    epochs_input: int = 80
    shuffle: bool = True
    verbose_trainer: int = 2

    ### Debugging parameter
    debug_print_mode: bool = False


    class OneCycleSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        """
        Three-phase One-Cycle schedule *à la* Shih-Pang, with optional
        momentum mirroring:

            warm-up        base_lr  → max_lr       (linear)
            cooldown       max_lr   → base_lr      (cosine)
            annihilation   base_lr  → final_lr     (linear, usually 10× lower)

        If `momentum=True` the returned value is an **inverse curve**
        ranging between `max_mom` and `min_mom`, so that momentum starts
        high, dips when LR peaks, and climbs back at the end.
        """

        def __init__(
            self,
            max_lr: float,
            steps_per_epoch: int,
            warmup_epochs: int,
            cooldown_epochs: int,
            annihilation_epochs: int,
            div_factor: float = 50.0,
            final_div: float = 10.0,
            *,
            momentum: bool = False,
            max_mom: float = 0.95,
            min_mom: float = 0.85,
            name: str = "OneCycleSchedule",
        ):
            super().__init__()
            # ---- phases (in steps) ------------------------------------------------
            self.ws = int(warmup_epochs * steps_per_epoch)
            self.cs = int(cooldown_epochs * steps_per_epoch)
            self.as_ = int(annihilation_epochs * steps_per_epoch)
            self.total_steps = self.ws + self.cs + self.as_

            # ---- LR levels --------------------------------------------------------
            self.max_lr   = float(max_lr)
            self.base_lr  = self.max_lr / div_factor      # paper: div_factor = 50
            self.final_lr = self.base_lr / final_div      # paper: final_div = 10

            # ---- optional momentum -----------------------------------------------
            self.momentum   = bool(momentum)
            self.max_mom    = float(max_mom)
            self.min_mom    = float(min_mom)
            self.name       = name

        # ------------------------------------------------------------------------- #
        #                                core logic                                 #
        # ------------------------------------------------------------------------- #
        def _lr_value(self, s: tf.Tensor) -> tf.Tensor:
            """Piece-wise LR value at scalar/array step `s` (float32)."""
            max_lr   = tf.cast(self.max_lr,   tf.float32)
            base_lr  = tf.cast(self.base_lr,  tf.float32)
            final_lr = tf.cast(self.final_lr, tf.float32)
            ws = tf.cast(self.ws,  tf.float32)
            cs = tf.cast(self.cs,  tf.float32)
            as_ = tf.cast(self.as_, tf.float32)

            # ---- phase 1: warm-up (linear) --------------------------------------
            lr_wu = base_lr + (max_lr - base_lr) * (s / ws)

            # ---- phase 2: cooldown (cosine) --------------------------------------
            cos_s     = tf.maximum(s - ws, 0.0)
            cos_decay = 0.5 * (1.0 + tf.cos(np.pi * cos_s / cs))
            lr_cd     = base_lr + (max_lr - base_lr) * cos_decay

            # ---- phase 3: annihilation (linear) ----------------------------------
            lin_s   = tf.maximum(s - ws - cs, 0.0)
            lr_an   = base_lr + (final_lr - base_lr) * (lin_s / as_)

            # ---- pick phase -------------------------------------------------------
            m1 = s < ws
            m2 = tf.logical_and(s >= ws, s < ws + cs)
            return tf.where(m1, lr_wu,
                            tf.where(m2, lr_cd, lr_an))

        def __call__(self, step):
            """Returns LR *or* momentum, depending on `self.momentum`."""
            s  = tf.cast(step, tf.float32)
            lr = self._lr_value(s)

            if not self.momentum:
                return lr                       # --- ordinary LR schedule ---------

            # ---- momentum = mirrored LR ------------------------------------------
            #   high momentum ↔ low LR,   low momentum ↔ high LR
            #
            # normalise LR into [0, 1] w.r.t. (final_lr .. max_lr) range,
            # then invert and scale to (min_mom .. max_mom).
            lr_norm = (lr - self.final_lr) / (self.max_lr - self.final_lr)
            mom     = self.max_mom - lr_norm * (self.max_mom - self.min_mom)
            return mom

        # convenience: numpy copy of the whole curve
        def get_curve(self) -> np.ndarray:
            steps = tf.range(self.total_steps, dtype=tf.float32)
            return self(steps).numpy()

        def get_config(self):                  # so the schedule is serialisable
            return {
                "max_lr": self.max_lr,
                "steps_per_epoch": 1,  # not used on deserialisation but required
                "warmup_epochs": self.ws,  # raw numbers suffice
                "cooldown_epochs": self.cs,
                "annihilation_epochs": self.as_,
                "div_factor": self.max_lr / self.base_lr,
                "final_div": self.base_lr / self.final_lr,
                "momentum": self.momentum,
                "max_mom": self.max_mom,
                "min_mom": self.min_mom,
                "name": self.name,
            }

    class OneCyclePlateau(tf.keras.optimizers.schedules.LearningRateSchedule):
        """
        LR(t) =
        1) linear warmup:    base -> max              (warmup_epochs)
        2) cosine rampdown:  max  -> plateau          (ramp_epochs)
        3) FLAT plateau:     plateau                  (plateau_epochs)   <-- NEW
        4) cosine cooldown:  plateau -> base          (cooldown_epochs)
        5) linear annihil.:  base -> final            (annihilation_epochs)

        If momentum=True, returns the mirrored momentum curve in [min_mom, max_mom].
        """

        def __init__(
            self,
            *,
            steps_per_epoch: int,
            warmup_epochs: int,
            ramp_epochs: int,       # max -> plateau (cosine)
            plateau_epochs: int,    # hold at plateau (flat)
            cooldown_epochs: int,   # plateau -> base (cosine)
            annihilation_epochs: int,
            base_lr: float,
            max_lr: float,
            plateau_lr: float,
            final_lr: float,
            momentum: bool = False,
            max_mom: float = 0.95,
            min_mom: float = 0.85,
            name: str = "OneCyclePlateau",
        ):
            super().__init__()
            # ---- steps per phase ----------------------------------------------------
            self.ws  = int(warmup_epochs        * steps_per_epoch)
            self.rs  = int(ramp_epochs          * steps_per_epoch)
            self.ps  = int(plateau_epochs       * steps_per_epoch)
            self.cs  = int(cooldown_epochs      * steps_per_epoch)
            self.as_ = int(annihilation_epochs  * steps_per_epoch)
            self.total_steps = self.ws + self.rs + self.ps + self.cs + self.as_

            # ---- levels -------------------------------------------------------------
            self.base_lr    = float(base_lr)
            self.max_lr     = float(max_lr)
            self.plateau_lr = float(plateau_lr)
            self.final_lr   = float(final_lr)

            # ---- momentum mirror ----------------------------------------------------
            self.momentum = bool(momentum)
            self.max_mom  = float(max_mom)
            self.min_mom  = float(min_mom)
            self.name     = name

        # guard for zero-length phases
        def _zdiv(self, num, den):
            return num / tf.maximum(den, 1.0)

        def _lr_value(self, s: tf.Tensor) -> tf.Tensor:
            base_lr    = tf.cast(self.base_lr,    tf.float32)
            max_lr     = tf.cast(self.max_lr,     tf.float32)
            plateau_lr = tf.cast(self.plateau_lr, tf.float32)
            final_lr   = tf.cast(self.final_lr,   tf.float32)

            ws  = tf.cast(self.ws,  tf.float32)
            rs  = tf.cast(self.rs,  tf.float32)
            ps  = tf.cast(self.ps,  tf.float32)
            cs  = tf.cast(self.cs,  tf.float32)
            as_ = tf.cast(self.as_, tf.float32)

            # phase 1: linear warmup base -> max
            lr_wu = base_lr + (max_lr - base_lr) * self._zdiv(s, ws)

            # phase 2: cosine rampdown max -> plateau
            s2 = tf.maximum(s - ws, 0.0)
            cos2 = 0.5 * (1.0 + tf.cos(np.pi * self._zdiv(s2, rs)))   # 1 -> 0
            lr_rd = plateau_lr + (max_lr - plateau_lr) * cos2

            # phase 3: FLAT plateau (constant)
            lr_pl = tf.fill(tf.shape(s), plateau_lr)

            # phase 4: cosine cooldown plateau -> base
            s4 = tf.maximum(s - ws - rs - ps, 0.0)
            cos4 = 0.5 * (1.0 + tf.cos(np.pi * self._zdiv(s4, cs)))   # 1 -> 0
            lr_cd = base_lr + (plateau_lr - base_lr) * cos4

            # phase 5: linear annihilation base -> final
            s5 = tf.maximum(s - ws - rs - ps - cs, 0.0)
            lr_an = base_lr + (final_lr - base_lr) * self._zdiv(s5, as_)

            # choose phase
            m1 = s < ws
            m2 = tf.logical_and(s >= ws,                s < ws + rs)
            m3 = tf.logical_and(s >= ws + rs,           s < ws + rs + ps)
            m4 = tf.logical_and(s >= ws + rs + ps,      s < ws + rs + ps + cs)
            return tf.where(m1, lr_wu,
                            tf.where(m2, lr_rd,
                                    tf.where(m3, lr_pl,
                                            tf.where(m4, lr_cd, lr_an))))

        def __call__(self, step):
            s = tf.cast(step, tf.float32)
            lr = self._lr_value(s)
            if not self.momentum:
                return lr

            # mirror LR into momentum range
            lr_min = tf.cast(self.final_lr, tf.float32)
            lr_max = tf.cast(self.max_lr,   tf.float32)
            lr_norm = (lr - lr_min) / tf.maximum(lr_max - lr_min, 1e-12)
            return self.max_mom - lr_norm * (self.max_mom - self.min_mom)

        def get_curve(self) -> np.ndarray:
            steps = tf.range(self.total_steps, dtype=tf.float32)
            return self(steps).numpy()

        def get_config(self):
            return {
                "steps_per_epoch": 1,
                "warmup_epochs": self.ws,
                "ramp_epochs": self.rs,
                "plateau_epochs": self.ps,
                "cooldown_epochs": self.cs,
                "annihilation_epochs": self.as_,
                "base_lr": self.base_lr,
                "max_lr": self.max_lr,
                "plateau_lr": self.plateau_lr,
                "final_lr": self.final_lr,
                "momentum": self.momentum,
                "max_mom": self.max_mom,
                "min_mom": self.min_mom,
                "name": self.name,
            }



    # ### scheduler init
    total_events = 1e4
    if MODE == 'A':
        coarse_voxels = 648
    else:
        coarse_voxels = 540
    single_fv_events = total_events * coarse_voxels
    single_fv_events_train = np.ceil(single_fv_events*0.7) 
    steps_per_epoch = np.ceil(single_fv_events_train / batch_size).astype(int)
    total_steps = steps_per_epoch * epochs_input

    warmup_epochs       = 17
    cooldown_epochs     = 59
    annihilation_epochs = 4
    max_lr              = 1e-3
    base_lr             = 5e-6
    div_factor          = max_lr / base_lr # so base_lr = 2e-5
    final_div           = 10.0 # so final_lr = 2e-6

    if warmup_epochs + cooldown_epochs + annihilation_epochs != epochs_input:
        raise ValueError(f"warmup_epochs + cooldown_epochs + annihilation_epochs must equal epochs_input, \
                         but got {warmup_epochs} + {cooldown_epochs} + {annihilation_epochs} != {epochs_input}")

    '''
    schedule_lr = OneCycleSchedule(max_lr            = max_lr,
                                steps_per_epoch      = steps_per_epoch,
                                warmup_epochs        = warmup_epochs,
                                cooldown_epochs      = cooldown_epochs,
                                annihilation_epochs  = annihilation_epochs,
                                div_factor           = div_factor,  
                                final_div            = final_div     
    )

    schedule_mom = OneCycleSchedule(1e-3, steps_per_epoch,
                                warmup_epochs       = warmup_epochs,
                                cooldown_epochs     = cooldown_epochs,
                                annihilation_epochs = annihilation_epochs,
                                momentum            = True,           # << flip!
                                max_mom=0.95, min_mom=0.85)

    '''

    schedule_lr = OneCyclePlateau(steps_per_epoch=steps_per_epoch,
                            warmup_epochs=10,          # base -> max
                            ramp_epochs=20,            # max -> plateau (cosine)
                            plateau_epochs=25,        # <-- flat
                            cooldown_epochs=20,       # plateau -> base (cosine)
                            annihilation_epochs=5,    # base -> final (linear)
                            base_lr=2e-5,
                            max_lr=1e-3,
                            plateau_lr=3e-4,
                            final_lr=2e-6,
                            momentum=False,
                            )
    

    schedule_mom = OneCyclePlateau(steps_per_epoch=steps_per_epoch,
                                warmup_epochs=10,          # base -> max
                                ramp_epochs=20,            # max -> plateau (cosine)
                                plateau_epochs=25,        # <-- flat
                                cooldown_epochs=20,       # plateau -> base (cosine)
                                annihilation_epochs=5,    # base -> final (linear)
                                base_lr=2e-5,
                                max_lr=1e-3,
                                plateau_lr=3e-4,
                                final_lr=2e-6,
                                momentum=True,
                                max_mom=0.95,
                                min_mom=0.88,
                                )
    

    # ## Trainer init
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    print(f"Number of devices: {num_gpus}")


    '''
    with strategy.scope():  
        NFObject: Trainer.Trainer = Trainer.Trainer(base_distribution = base_dist,
                                                flow = MAF, 
                                                x_data_train = x_train_ds,
                                                y_data_train = y_train_ds,
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
                                                                                        x_train = x_train_ds,
                                                                                        y_train = y_train_ds,
                                                                                        ndims = ndims,
                                                                                        total_steps = total_steps,
                                                                                        warmup_epochs = warmup_epochs,
                                                                                        cooldown_epochs = cooldown_epochs,
                                                                                        annihilation_epochs = annihilation_epochs,),
                                                fit_kwargs = get_fit_kwargs(batch_size = batch_size,
                                                                            epochs_input = epochs_input,
                                                                            validation_data = (x_val_ds, y_val_ds),
                                                                            shuffle = shuffle,
                                                                            verbose = verbose_trainer),
                                                debug_print_mode = debug_print_mode)
    '''

    NFObject: Trainer.Trainer = Trainer.Trainer(base_distribution = base_dist,
                                            flow = MAF, 
                                            x_data_train = x_train_ds,
                                            y_data_train = y_train_ds,
                                            io_kwargs = get_io_kwargs(path_to_results = path_to_results),
                                            compiler_kwargs = get_compiler_kwargs(lr = schedule_lr,
                                                                                    ignore_nans = True,
                                                                                    nan_threshold = nan_threshold,
                                                                                    beta_1_scheduler = schedule_mom,),
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
                                                                                    x_train = x_train_ds,
                                                                                    y_train = y_train_ds,
                                                                                    ndims = ndims,
                                                                                    total_steps = total_steps,
                                                                                    warmup_epochs = warmup_epochs,
                                                                                    cooldown_epochs = cooldown_epochs,
                                                                                    annihilation_epochs = annihilation_epochs,),
                                            fit_kwargs = get_fit_kwargs(batch_size = batch_size,
                                                                        epochs_input = epochs_input,
                                                                        validation_data = (x_val_ds, y_val_ds),
                                                                        shuffle = shuffle,
                                                                        verbose = verbose_trainer),
                                            debug_print_mode = debug_print_mode)
    
    trainable_params: int = NFObject.trainable_params
    non_trainable_params: int = NFObject.non_trainable_params


    # ## Store/save hyperparameters
    ### Initialize dictionaries to store paramerers and results ###
    results_dict: Dict[str, Any] = Utils.init_results_dict()
    hyperparams_dict: Dict[str, Any] = Utils.init_hyperparams_dict()

    ### Create log file ###
    log_file_name: str = Utils.create_log_file(mother_output_dir, results_dict)


    ## data is organized in datasets, the number of samples is a bit difficult to access, need to be infered from the cache file
    nsamples_train: int = 10_000 
    nsamples_val: int = 0
    nsamples_test: int = 0  # No test set in this example

    ## seeds not used in this example, but can be set for reproducibility
    seed_train: int = 42
    seed_test: int = 42
    seed_dist: int = 42
    seed_metrics: int = 42

    hyperparams_dict = Utils.update_hyperparams_dict(hyperparams_dict = hyperparams_dict,
                                                    run_number = run_number,
                                                    n_runs = n_runs,
                                                    seeds = [seed_train, seed_test, seed_dist, seed_metrics],
                                                    nsamples = [nsamples_train, nsamples_val, nsamples_test],
                                                    ndims = ndims,
                                                    corr = None,
                                                    bijector_name = bijector_name,
                                                    nbijectors = num_bijectors,
                                                    spline_knots = spline_knots,
                                                    range_min = range_min,
                                                    hllabel = '-'.join(str(e) for e in n_hidden),
                                                    trainable_parameters = trainable_params,
                                                    non_trainable_parameters = non_trainable_params,
                                                    batch_size = batch_size,
                                                    epochs_input = epochs_input,
                                                    activation = activation,
                                                    regulariser = regulariser,
                                                    eps_regulariser = eps_regulariser,
                                                    training_device = training_device)
    Utils.save_hyperparams_dict(path_to_results, hyperparams_dict)


    # # Train
    NFObject.train()
    print("Training completed.")


if __name__ == "__main__":
    main()
