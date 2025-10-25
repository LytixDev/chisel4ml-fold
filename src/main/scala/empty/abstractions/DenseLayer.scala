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
                       output: TensorSpec,
                       mulDt: IntegerDataType, // Data type for multiplication results
                       accDt: IntegerDataType, // Data type for accumulators
                       //bias: Option[TensorData] = None,
                       activationFunc: ActivationFunc = Identity,
                       PEsPerOutput: Int
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
  require(PEsPerOutput >= 1 && PEsPerOutput <= n)
  require(n % PEsPerOutput == 0)
}