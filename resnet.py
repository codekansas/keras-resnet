from __future__ import absolute_import

import operator

from keras import backend as K
from keras.engine import InputSpec
from keras.layers import initializations, Wrapper, merge


class Residual(Wrapper):
  """This wrapper automatically applies a residual to a model.

  For an input `x` and a model `F(x)`, the residual wrapper gives the output
  `y = x + F(x)`. In this configuration, the output of F(x) must have the
  same shape as x. Setting `merge_mode='weighted'` defines an additional vector
  `U` such that the output becomes a partition between x and F(x), i.e.
  `y = U * x + (1 - U) * F(x)`. Other merge modes are supported, with their
  corresponding arguments passed as keyword arguments.

  The wrapper can be applied to give any model a residual. For example:

  ```python
      input = Input(shape=(5,))

      # Apply the residual normally
      output1 = Residual(Dense(5))(input)

      # Throws an exception due to mismatching shapes
      output2 = Residual(Dense(3))(input)
  ```

  Arguments:
      layer: The layer to wrap
      merge_mode: Like regular merge function, but with extra option 'weighted'
  """
  def __init__(self, layer, merge_mode='sum', **merge_params):
    self.merge_mode = merge_mode
    self.merge_params = merge_params
    self.supports_masking = True
    super(Residual, self).__init__(layer)

  def build(self, input_shape):
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
    input_shape = self.input_spec[0].shape
    output_shape = self.layer.get_output_shape_for(input_shape)
    if output_shape != input_shape:
      raise Exception('Cannot apply residual to layer "{}": '
                      'mismatching input and output shapes'
                      '"{}" and "{}"'
                      .format(self.layer.name, input_shape, output_shape))
    layer_output = self.layer.call(x, mask)
    if self.merge_mode == 'weighted':
      output = x * self.U + layer_output * (1 - self.U)
    else:
      output = merge([x, layer_output], self.merge_mode, **self.merge_params)
    return output
