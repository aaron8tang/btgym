import warnings
warnings.filterwarnings("ignore") # suppress h5py deprecation warning

import os
import backtrader as bt
import numpy as np

from btgym import BTgymEnv, BTgymDataset
from btgym.strategy.observers import Reward, Position, NormPnL
from btgym.algorithms import Launcher, Unreal, AacStackedRL2Policy
from btgym.research import DevStrat_4_11

# Handy function:
def under_the_hood(env):
    """Shows environment internals."""
    for attr in ['dataset','strategy','engine','renderer','network_address']:
        print ('\nEnv.{}: {}'.format(attr, getattr(env, attr)))

    for params_name, params_dict in env.params.items():
        print('\nParameters [{}]: '.format(params_name))
        for key, value in params_dict.items():
            print('{} : {}'.format(key,value))
# Set backtesting engine parameters:

MyCerebro = bt.Cerebro()

# Define strategy and broker account parameters:
MyCerebro.addstrategy(
    DevStrat_4_11,
    start_cash=2000,  # initial broker cash
    commission=0.0001,  # commisssion to imitate spread
    leverage=10.0,
    order_size=2000,  # fixed stake, mind leverage
    drawdown_call=10, # max % to loose, in percent of initial cash
    target_call=10,  # max % to win, same
    skip_frame=10,
    gamma=0.99,
    reward_scale=7, # gardient`s nitrox, touch with care!
    state_ext_scale = np.linspace(3e3, 1e3, num=5)
)
# Visualisations for reward, position and PnL dynamics:
MyCerebro.addobserver(Reward)
MyCerebro.addobserver(Position)
MyCerebro.addobserver(NormPnL)

# Data: uncomment to get up to six month of 1 minute bars:

# Uncomment single choice:
MyDataset = BTgymDataset(
    filename='./data/sh601318.csv',  # simple sine
    start_weekdays={0, 1, 2, 3, 4},
	episode_duration={'days': 90, 'hours': 0, 'minutes': 0},
	# Want to start every episode at the begiining of the day:
	start_00=True,
    time_gap={'days': 1},
)

env_config = dict(
    class_ref=BTgymEnv,
    kwargs=dict(
        dataset=MyDataset,
        engine=MyCerebro,
        render_modes=['episode', 'human', 'internal', ], #'external'],
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
    num_workers=20,  # set according CPU's available or so
    num_ps=1,
    num_envs=1,
    log_dir=os.path.expanduser('~/tmp/test_4_11'),
)

policy_config = dict(
    class_ref=AacStackedRL2Policy,
    kwargs={
        'lstm_layers': (256, 256),
        'lstm_2_init_period': 60,
    }
)

trainer_config = dict(
    class_ref=Unreal,
    kwargs=dict(
        opt_learn_rate=[1e-4, 1e-4], # random log-uniform
        opt_end_learn_rate=1e-5,
        opt_decay_steps=50*10**6,
        model_gamma=0.99,
        model_gae_lambda=1.0,
        model_beta=0.05, # entropy reg
        rollout_length=20,
        time_flat=True,
        use_value_replay=False,
        model_summary_freq=10,
        episode_summary_freq=1,
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
    verbose=1
)

# Train it:
launcher.run()