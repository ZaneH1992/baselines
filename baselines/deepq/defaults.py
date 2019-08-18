def atari():
    return dict(
        network='conv_only',
        lr=1e-4,
        buffer_size=1e6,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=80000,
        target_network_update_freq=40000,
        gamma=0.99,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=10000,
        checkpoint_path=None,
        dueling=False
    )

def retro():
    return atari()

