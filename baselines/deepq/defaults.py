import tensorflow as tf
def atari():
    return dict(
        network='conv_only',
        lr=1e-4,
        buffer_size=int(1e5),
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=80000,
        target_network_update_freq=40000,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=100000,
        checkpoint_path=None,
        dueling=False,
        weight_initializer=tf.variance_scaling_initializer(scale=2)
    )

def retro():
    return atari()

