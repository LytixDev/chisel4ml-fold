package empty.abstractions

import chisel3._

case class IntegerDataType(
                        bitWidth: Int,
                        isSigned: Boolean,
                      )

// Tensor Specification
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

sealed trait Activation
case object ReLU extends Activation
case object Sigmoid extends Activation
case object Identity extends Activation

case class DenseLayer(
                       input: TensorSpec,
                       weights: TensorData,
                       output: TensorSpec,
                       mulDt: IntegerDataType, // Data type for multiplication results
                       accDt: IntegerDataType, // Data type for accumulators
                       //bias: Option[TensorData] = None,
                       activation: Activation = Identity,
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

//case class QTensor(rows: Int, // m
//                   cols: Int, // n
//                   data: Array[Array[Int]],
//                   hasData: Boolean,
//                   // Uniform quantization parameters
//                   shamt: Int, // NOTE: shift amount for power of two scaling
//                   //zeroPoint: Int, // NOTE: not supported yet
//                   )

// TODO: functions that return expectedLatency, expectedAreaCost, ...
/*
 * input matrix: m*n
 * weight matrix: n*k
 * PEsPerOutput: number of multipliers per output (1 = fully folded, n = fully parallel)
 */
//case class DenseLayer(m: Int, n: Int, k: Int, weights: Array[Array[Int]], PEsPerOutput: Int, quantizationScheme: QuantizationScheme) {

// TODO:
//  - want to get rid of the quantizationScheme and rather bake this into the QTensors
//  - activation function
//  - bias/thresholding
//  - we need the final shamt amount, probably we need to take in an out tensor as well
//  Come to think of it, I think I prefer to just write all of the params out?
// case class DenseLayer(in: QTensor, weights: QTensor, PEsPerOutput: Int, quantizationScheme: QuantizationScheme) {
  // TODO:
  // require(weights.length == n, s"weights must have $n rows, got ${weights.length}")
  // require(weights.forall(_.length == k), s"all weight rows must have $k columns")
  // require(1 <= PEsPerOutput && PEsPerOutput <= n, s"PEsPerOutput must be between 1 (fully folded) and $n (fully parallel)")
  // require(n % PEsPerOutput == 0, s"n=$n must be divisible by PEsPerOutput=$PEsPerOutput")
// }

// info about:
// - input:
//   - dimensions
//   - shamt
//   - bit width
//   - is signed
// - weights:
//   - dimensions
//   - actual data
//   - shamt
//   - bit width
//   - is signed
// - bias:
//   - dimensions (can be inferred)
//   - actual data
//   - shamt
//   - bit width
//   - is signed
// - output:
//   - dimensions (can be inferred)
//   - bid width
//   - shamt
//   - is signed
// - activation function
// - PEsPerOutput
// - bit width and signedness of multiplication result
// - bit width and signedness of accumulators

// def __init__(
//               self,
//               in_features: int,
//               out_features: int,
//               bias: Optional[bool] = True,
//               weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
//               bias_quant: Optional[BiasQuantType] = None,
//               input_quant: Optional[ActQuantType] = None,
//               output_quant: Optional[ActQuantType] = None,
//               return_quant_tensor: bool = False,
//               device: Optional[torch.device] = None,
//               dtype: Optional[torch.dtype] = None,
//               **kwargs) -> None:
//