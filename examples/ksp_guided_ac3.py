import warnings
warnings.filterwarnings("ignore") # suppress h5py deprecation warning

import os
import backtrader as bt
import numpy as np
from gym import spaces

from btgym import BTgymEnv, BTgymDataset, BTgymRandomDataDomain
from btgym.algorithms import Launcher

from btgym.research.gps.aac import GuidedAAC
from btgym.research.gps.policy import GuidedPolicy_0_0
from btgym.research.gps.strategy import GuidedStrategy_0_0, ExpertObserver

# Set backtesting engine and parameters:

engine = bt.Cerebro()

engine.addstrategy(
    GuidedStrategy_0_0,
    drawdown_call=10, # max % to loose, in percent of initial cash
    target_call=10,  # max % to win, same
    skip_frame=10,
    gamma=0.99,
    state_ext_scale=np.linspace(4e3, 1e3, num=6),
    reward_scale=7,
    expert_config=  # see btgym.research.gps.oracle.Oracle class for details
        {
            'time_threshold': 5,
            'pips_threshold': 10,
            'pips_scale': 1e-4,
            'kernel_size': 10,
            'kernel_stddev': 1,
        },
)

# Expert actions observer:
engine.addobserver(ExpertObserver)

# Set leveraged account:
engine.broker.setcash(2000)
engine.broker.setcommission(commission=0.0001, leverage=10.0) # commisssion to imitate spread
engine.addsizer(bt.sizers.SizerFix, stake=5000)

# Data: uncomment to get up to six month of 1 minute bars:
data_m1_6_month = [
    './data/DAT_ASCII_EURUSD_M1_201701.csv',
    './data/DAT_ASCII_EURUSD_M1_201702.csv',
    './data/DAT_ASCII_EURUSD_M1_201703.csv',
    #'./data/DAT_ASCII_EURUSD_M1_201704.csv',
    #'./data/DAT_ASCII_EURUSD_M1_201705.csv',
    #'./data/DAT_ASCII_EURUSD_M1_201706.csv',
]


# Uncomment single choice of source file:
dataset = BTgymRandomDataDomain(
    #filename=data_m1_6_month,
    #filename='./data/DAT_ASCII_EURUSD_M1_2016.csv', # full year
    filename='/home/aaron8tang/projects/btgym/examples/data/sh601318.csv',  # simple sine
    parsing_params=dict(
        header=None,
    ),
    trial_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 1, 'hours': 0, 'minutes': 0},
        start_00=False,
        time_gap={'days': 1, 'hours': 0},
        test_period={'days': 5, 'hours': 0, 'minutes': 0},
    ),
    episode_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 1, 'hours': 0, 'minutes': 0},
        start_00=False,
        time_gap={'days': 1, 'hours': 0},
    ),
)

env_config = dict(
    class_ref=BTgymEnv,
    kwargs=dict(
        dataset=dataset,
        engine=engine,
        render_modes=['episode', 'human', 'external', 'internal'],
        render_state_as_image=True,
        render_ylabel='OHL_diff. / Internals',
        render_size_episode=(12,8),
        render_size_human=(9, 4),
        render_size_state=(11, 3),
        render_dpi=75,
        port=5000,
        data_port=4999,
        connect_timeout=90,
        verbose=1,
    )
)

cluster_config = dict(
    host='127.0.0.1',
    port=12230,
    num_workers=1,  # Set according CPU's available or so
    num_ps=1,
    num_envs=1,
    log_dir=os.path.expanduser('~/tmp/gps'),
)

policy_config = dict(
    class_ref=GuidedPolicy_0_0,
    kwargs={
        'lstm_layers': (256, 256),
        'lstm_2_init_period': 50,
        'conv_2d_layer_config': (
             (32, (3, 1), (2, 1)),
             (32, (3, 1), (2, 1)),
             (64, (3, 1), (2, 1)),
             (64, (3, 1), (2, 1))
         ),
        'encode_internal_state': False,
    }
)

trainer_config = dict(
    class_ref=GuidedAAC,
    kwargs=dict(
        opt_learn_rate=1e-4, # scalar or random log-uniform
        opt_end_learn_rate=1e-5,
        opt_decay_steps=20*10**6,
        model_gamma=0.99,
        model_gae_lambda=1.0,
        model_beta=0.01, # Entropy reg, scalar or random log-uniform
        aac_lambda=1.0, # main a3c loss weight
        guided_lambda=1.0,  # Imitation loss weight
        guided_decay_steps=10*10**6,  # annealing guided_lambda to zero in 10M steps
        rollout_length=20,
        time_flat=True,
        use_value_replay=False,
        episode_train_test_cycle=[1,0],
        model_summary_freq=100,
        episode_summary_freq=5,
        env_render_freq=5,
    )
)

launcher = Launcher(
    cluster_config=cluster_config,
    env_config=env_config,
    trainer_config=trainer_config,
    policy_config=policy_config,
    test_mode=False,
    max_env_steps=100*10**6,
    root_random_seed=0,
    purge_previous=1,  # ask to override previously saved model and logs
    verbose=1,
)

# Train it:
launcher.run()