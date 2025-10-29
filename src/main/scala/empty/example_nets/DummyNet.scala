package empty.example_nets

import empty.abstractions.{DenseLayer, IntegerDataType, TensorData, TensorSpec}

object DummyNet {
  def apply(): Array[DenseLayer] = {
    // Layer 1: 1x4 @ 4x2, with 1 PEs for each output (takes 4 cycles)
    // Layer 2: 1x2 @ 2x1, with 2 PEs for each output (takes 1 cycle)

    val weights1 = Array(
      Array(1, 0),
      Array(0, 1),
      Array(1, 1),
      Array(0, 1)
    )
    val weights2 = Array(
      Array(2),
      Array(3)
    )

    // Layer 1
    val in1Spec = TensorSpec(
      rows = 1, cols = 4,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val w1Data = TensorData(
      spec = TensorSpec(
        rows = 4, cols = 2,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      ),
      data = weights1
    )
    val out1Spec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val layer1 = DenseLayer(
      input = in1Spec,
      weights = w1Data,
      output = out1Spec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
      accDt = IntegerDataType(bitWidth = 32, isSigned = false),
      PEsPerOutput = 1
    )

    // Layer 2
    val in2Spec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val w2Data = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 1,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      ),
      data = weights2
    )
    val out2Spec = TensorSpec(
      rows = 1, cols = 1,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val layer2 = DenseLayer(
      input = in2Spec,
      weights = w2Data,
      output = out2Spec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
      accDt = IntegerDataType(bitWidth = 32, isSigned = false),
      PEsPerOutput = 2
    )

    Array(layer1, layer2)
  }
}
