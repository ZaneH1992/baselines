from baselines.rdqn import models  # noqa
from baselines.rdqn.build_graph import build_act, build_train  # noqa
from baselines.rdqn.rdqn import learn, load_act  # noqa
from baselines.rdqn.replay_buffer import ReplayBuffer, ActionreplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)