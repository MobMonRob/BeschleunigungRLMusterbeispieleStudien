class MultiheadActor(nn.Module):
    def __init__(self, state_net, num_heads=10):
        super().__init__()
        self.state_net = state_net
        self.num_heads = num_heads
        self.heads = nn.ModuleList([nn.Linear(state_net.output_dim, state_net.action_dim) for _ in range(num_heads)])

    def forward(self, state):
        features = self.state_net(state)
        actions = torch.stack([head(features) for head in self.heads], dim=1)
        return actions

class UR5Actor(ActorSpec):
    ...
    @classmethod
    def get_from_params(
        cls,
        state_net_params: Dict,
        policy_head_params: Dict,
        env_spec: EnvironmentSpec,
    ):
        ...
        state_net = StateNet.get_from_params(im_width, im_height, in_channels, **state_net_params)
        actor_net = MultiheadActor(state_net, num_heads=10)

        net = cls(state_net=actor_net, head_net=None)
        print('-----------Actor------------\n{}'.format(net))
        return net
