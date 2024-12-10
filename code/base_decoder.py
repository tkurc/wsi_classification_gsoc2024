
class BaseDecoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def validate_input_shape(self, foundation_output_shape):
        if self.input_shape != "any" and self.input_shape != foundation_output_shape:
            raise ValueError("Input shape mismatch with foundation model output")

class CustomDecoder(BaseDecoder):
    def decode(self, features):
        # Decoding logic
        pass
