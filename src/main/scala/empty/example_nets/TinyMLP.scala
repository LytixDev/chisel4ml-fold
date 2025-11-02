package empty.example_nets

import empty.abstractions.{DenseLayer, IntegerDataType, ReLU, TensorData, TensorSpec}

object TinyMLP {
  def apply(): Array[DenseLayer] = {
    // l1: in: 1x16, weights: 16 x 16, out: 1x16
    // l2: in: 1x16, weights: 16x1, out: 1x1

    val rand = new scala.util.Random(42)
    val in1 = TensorSpec(
      rows = 1, cols = 16,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = -1
    )
    val weights1 = TensorData(
      spec = TensorSpec(
        rows = 16, cols = 16,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 2
      ),
      data = Array.fill(16, 16)(rand.nextInt(8) - 4)
    )
    val out1 = TensorSpec(
      rows = 1, cols = 16,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = -3
    )

    val layer1 = DenseLayer(
      input = in1,
      weights = weights1,
      output = out1,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = true),
      accDt = IntegerDataType(bitWidth = 32, isSigned = true),
      activationFunc = ReLU,
      PEsPerOutput = 1
    )

    val in2 = TensorSpec(
      rows = 1, cols = 16,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = -3
    )
    val weights2 = TensorData(
      spec = TensorSpec(
        rows = 16, cols = 1,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 2
      ),
      data = Array.fill(16, 1)(rand.nextInt(8) - 4)
    )
    val out2 = TensorSpec(
      rows = 1, cols = 1,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 1
    )

    val layer2 = DenseLayer(
      input = in2,
      weights = weights2,
      output = out2,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = true),
      accDt = IntegerDataType(bitWidth = 32, isSigned = true),
      PEsPerOutput = 1
    )

    Array(layer1, layer2)
  }
}
