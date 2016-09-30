from __future__ import absolute_import

import operator

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import initializations, Wrapper, merge


class Residual(Wrapper):
    """This wrapper automatically applies a residual to a model.

    For an input `x` and a model `F(x)`, the residual wrapper gives the output
    `y = x + F(x)`. In this configuration, the output of F(x) must have the
    same shape as x. Other merge modes are supported besides summation.

    ```python
        input = Input(shape=(5,))

        # Apply the residual normally
        output1 = Residual(Dense(5), mode='sum')(input)

        # Throws an exception due to mismatching shapes
        output2 = Residual(Dense(3), mode='sum')(input)

        # Product: `y = x * F(x)`
        output3 = Residual(Dense(5), mode='mul')(input)
    ```

    For more modes, see: https://keras.io/layers/core/#merge

    Alternatively, a function which takes the input and the layer output
    can be passed to define the merge:

    ```python
        from keras.layers import merge
        def diff_merge(x, fx):
            diff = lambda x: x[1] - x[0]
            return merge([x, fx], mode=diff, output_shape=lambda x: x)

        # Difference: `y = F(x) - x`
        output4 = Residual(Dense(5), mode=diff_merge)(input)
    ```

    Arguments:
        layer: The layer to wrap
        merge_mode: The merge operation
        merge_params: Extra keyword arguments to pass to the merge function
    """
    def __init__(self, layer, mode='sum', **merge_params):
        self.merge_mode = mode
        self.merge_params = merge_params
        self.supports_masking = True
        super(Residual, self).__init__(layer)

    def build(self, input_shape):
        output_shape = self.layer.get_output_shape_for(input_shape)
        if output_shape != input_shape:
            raise Exception('Cannot apply residual to layer "{}": '
                            'mismatching input and output shapes'
                            '"{}" and "{}"'
                            .format(self.layer.name, input_shape, output_shape))
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.input_spec = [InputSpec(shape=input_shape)]
        output_shape = self.layer.get_output_shape_for(input_shape)
        if self.merge_mode == 'weighted':
            self.U = K.random_uniform_variable(output_shape[1:], 0, 1,
                                               name='{}_U'.format(self.name))
        super(Residual, self).build()

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        layer_output = self.layer.call(x, mask)
        if isinstance(self.merge_mode, str):
            output = merge([x, layer_output], self.merge_mode, **self.merge_params)
        else:
            output = self.merge_mode(x, layer_output)
        return output
