class ProbabilisticCritic(nn.Module):
    def __init__(self, state_action_net):
        super().__init__()
        self.state_action_net = state_action_net

    def forward(self, state, action):
        features = self.state_action_net(state, action)
        return features

class UR5StateActionCritic(CriticSpec):
    ...
    @classmethod
    def get_from_params(
        cls,
        state_action_net_params: Dict,
        value_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        ...
        state_action_net = StateActionNet.get_from_params(im_width, im_height, in_channels, action_in_features, **state_action_net_params)
        critic_net = ProbabilisticCritic(state_action_net)

        net = cls(state_action_net=critic_net, head_net=None)
        print('---------Critic--------\n{}'.format(net))
        return net
