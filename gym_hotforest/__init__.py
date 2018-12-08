from gym.envs.registration import register

register(
    id='hotforest-v0',
    entry_point='gym_hotforest.envs:HotforestEnv',
)

