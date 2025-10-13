package empty.abstractions

import chisel3._

// TODO: functions that return expectedLatency, expectedAreaCost, ...
/*
 * input matrix: m*n
 * weight matrix: n*k
 * PEsPerOutput: number of multipliers per output (1 = fully folded, n = fully parallel)
 */
case class DenseLayer(m: Int, n: Int, k: Int, weights: Array[Array[Int]], PEsPerOutput: Int, neuronCompute: NeuronCompute) {
  require(weights.length == n, s"weights must have $n rows, got ${weights.length}")
  require(weights.forall(_.length == k), s"all weight rows must have $k columns")
  require(1 <= PEsPerOutput && PEsPerOutput <= n, s"PEsPerOutput must be between 1 (fully folded) and $n (fully parallel)")
  require(n % PEsPerOutput == 0, s"n=$n must be divisible by PEsPerOutput=$PEsPerOutput")
}