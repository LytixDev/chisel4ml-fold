package empty.abstractions

case class IntegerDataType(
                        bitWidth: Int,
                        isSigned: Boolean,
                      )

case class TensorSpec(
                       rows: Int,
                       cols: Int,
                       dt: IntegerDataType,
                       shamt: Int, // power-of-2 scaling
                       zeroPoint: Int = 0 // TODO: not implemented
                     )

case class TensorData(
                       spec: TensorSpec,
                       data: Array[Array[Int]]
                     ) {
  def rows = spec.rows
  def cols = spec.cols
  def dt = spec.dt
}

sealed trait ActivationFunc
case object ReLU extends ActivationFunc
case object Sigmoid extends ActivationFunc
case object Identity extends ActivationFunc

case class DenseLayer(
                       input: TensorSpec,
                       weights: TensorData,
                       bias: Option[TensorData] = None,
                       output: TensorSpec,
                       mulDt: IntegerDataType, // Data type for multiplication results
                       accDt: IntegerDataType, // Data type for accumulators
                       activationFunc: ActivationFunc = Identity,
                       multipliersPerDotProduct: Int
                     ) {
  // Convenience accessors
  def m = input.rows
  def n = input.cols
  def k = weights.cols

  // Validation
  require(weights.rows == n, s"Weight rows must match input cols")
  require(output.cols == k, s"Output cols must match weight cols")
  require(output.rows == m, s"Output rows must match input rows")
  //bias.foreach(b => require(b.cols == k && b.rows == 1, s"Bias must be 1x$k"))
  require(multipliersPerDotProduct >= 1 && multipliersPerDotProduct <= n)
  require(n % multipliersPerDotProduct == 0)
}

object DenseLayer {
  /* Create some dummy data for testing. Do not use for anything else ! */
  def withRandomWeights(
                         m: Int,
                         n: Int,
                         k: Int,
                         multipliersPerOutputElement: Int,
                         inputBitWidth: Int = 8,
                         weightBitWidth: Int = 4,
                         outputBitWidth: Int = 8,
                         mulBitWidth: Int = 16,
                         accBitWidth: Int = 32,
                         withBias: Boolean = false,
                         activationFunc: ActivationFunc = Identity,
                         seed: Int = 42
  ): DenseLayer = {
    val rand = new scala.util.Random(seed)

    val inputSpec = TensorSpec(
      rows = m, cols = n,
      dt = IntegerDataType(bitWidth = inputBitWidth, isSigned = false),
      shamt = 0
    )

    // Generate random weights in appropriate range
    val weightRange = 1 << (weightBitWidth - 1)
    val weightsData = TensorData(
      spec = TensorSpec(
        rows = n, cols = k,
        dt = IntegerDataType(bitWidth = weightBitWidth, isSigned = true),
        shamt = 0
      ),
      data = Array.fill(n, k)(rand.nextInt(weightRange * 2) - weightRange)
    )

    val biasData = if (withBias) {
      val biasRange = 1 << (weightBitWidth - 1)
      Some(TensorData(
        spec = TensorSpec(
          rows = 1, cols = k,
          dt = IntegerDataType(bitWidth = accBitWidth, isSigned = true),
          shamt = 0
        ),
        data = Array.fill(1, k)(rand.nextInt(biasRange * 2) - biasRange)
      ))
    } else {
      None
    }

    val outputSpec = TensorSpec(
      rows = m, cols = k,
      dt = IntegerDataType(bitWidth = outputBitWidth, isSigned = false),
      shamt = 0
    )

    DenseLayer(
      input = inputSpec,
      weights = weightsData,
      bias = biasData,
      output = outputSpec,
      mulDt = IntegerDataType(bitWidth = mulBitWidth, isSigned = true),
      accDt = IntegerDataType(bitWidth = accBitWidth, isSigned = true),
      activationFunc = activationFunc,
      multipliersPerDotProduct = multipliersPerOutputElement
    )
  }
}