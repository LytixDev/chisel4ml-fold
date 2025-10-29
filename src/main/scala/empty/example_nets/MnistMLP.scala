package empty.example_nets

import empty.abstractions.{DenseLayer, IntegerDataType, ReLU, TensorData, TensorSpec}

object MnistMLP {
  def apply(): Array[DenseLayer] = {
    // MNIST MLP with random weights
    // l1: in: 1x784, weights: 784 x 32, out: 1x32
    // l2: in: 1x32, weights: 32x10, out: 1x10

    val rand = new scala.util.Random(42)
    val in1 = TensorSpec(
      rows = 1, cols = 784,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val weights1 = TensorData(
      spec = TensorSpec(
        rows = 784, cols = 32,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 0
      ),
      data = Array.fill(784, 32)(rand.nextInt(8) - 4)  // Random weights in [-4, 3]
    )
    val bias1 = TensorData(
      spec = TensorSpec(
        rows = 1, cols = 32,
        dt = IntegerDataType(bitWidth = 32, isSigned = true),
        shamt = 0
      ),
      data = Array.fill(1, 32)(rand.nextInt(16) - 8)  // Random biases in [-8, 7]
    )
    val out1 = TensorSpec(
      rows = 1, cols = 32,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )

    val layer1 = DenseLayer(
      input = in1,
      weights = weights1,
      bias = Some(bias1),
      output = out1,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = true),
      accDt = IntegerDataType(bitWidth = 32, isSigned = true),
      activationFunc = ReLU,
      PEsPerOutput = 56  // 56
    )

    val in2 = TensorSpec(
      rows = 1, cols = 32,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val weights2 = TensorData(
      spec = TensorSpec(
        rows = 32, cols = 10,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 0
      ),
      data = Array.fill(32, 10)(rand.nextInt(8) - 4)
    )
    val bias2 = TensorData(
      spec = TensorSpec(
        rows = 1, cols = 10,
        dt = IntegerDataType(bitWidth = 32, isSigned = true),
        shamt = 0
      ),
      data = Array.fill(1, 10)(rand.nextInt(16) - 8)  // Random biases in [-8, 7]
    )
    val out2 = TensorSpec(
      rows = 1, cols = 10,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )

    val layer2 = DenseLayer(
      input = in2,
      weights = weights2,
      bias = Some(bias2),
      output = out2,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = true),
      accDt = IntegerDataType(bitWidth = 32, isSigned = true),
      PEsPerOutput = 2
    )

    Array(layer1, layer2)
  }
}
