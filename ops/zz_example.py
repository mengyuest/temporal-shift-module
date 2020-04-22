from ops.zz_arnet import ARNet

class ARNetWrapper:
    def __init__(self, model, state):
        self.state = state
        self.model = model
        self.model.rnn.register_forward_hook(self.policy_hook)

    def policy_hook(self, input, output):
        self.state.policy = argmax(output)

    def __call__(self, x):
        return self.model(x)

#TODO is this similar class as argparser?
class ARNetComponentConfig(ComponentConfig):
    some_params: str = ''

#TODO what data type can we use here? can it be numpy, or list?
class ARNetComponentState(ComponentState):
    policy: int
    other_state: float

class ARNetComponent(IOComponent):
    def __init__(self, config, state, input_stream, output_stream):
        super(ARNetComponent, self).__init__(config, input_stream, output_stream)

        self.state = state
        model = ARNet(config)
        self.arnet = ARNetWrapper(model, self.state)

    def process(self, data):
        return self.arnet(data)

#TODO how to do data transform?

input_stream = get_input_stream('kafka')
output_stream = get_output_stream('kafka')
config = ARNetComponentConfig()
state = ARNetComponentState()
component = ARNetComponent(config, state, input_stream, output_stream)

component.run()