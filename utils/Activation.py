from tensorflow.keras.layers import Layer, ReLU

class HardSigmoid(Layer):
    def __init__(self,name="HardSigmoid", **kwargs):
        super(HardSigmoid, self).__init__(name=name, **kwargs)
        self.relu6 = ReLU(6.)
    def call(self, inputs, **kwargs):
        x = self.relu6(inputs + 3) * (1. / 6)
        return x
    def get_config(self):
        config = {"relu6":self.relu6}
        base_config = super(HardSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class HardSwish(Layer):
    def __init__(self,name="HardSwish", **kwargs):
        super(HardSwish, self).__init__(name=name, **kwargs)
        self.hard_sigmoid = HardSigmoid()
    def call(self, inputs, **kwargs):
        x = self.hard_sigmoid(inputs) * inputs
        return x
    def get_config(self):
        config = {"hard_sigmoid":self.hard_sigmoid}
        base_config = super(HardSwish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))