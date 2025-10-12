package empty

import chisel3._


// Stripped down NeuronCompute abstraction from Chisel4ml
// This abstraction allows the same layer logic to work with different numeric types.
// - signed and unsigned ints, fixed point?, floating point?
// - custom quantization schemes
abstract class NeuronCompute {
  type I <: Bits // Input type
  type W <: Bits // Weight type
  type M <: Bits // Multiplication result type
  type A <: Bits // Accumulator type
  type O <: Bits // Output type

  // Generator methods: Chisel needs concrete instances (not just the types) to clone
  // when creating Vecs, Regs, etc. These methods provide those instances.
  def genI: I
  def genW: W
  def genM: M
  def genA: A
  def genO: O

  // def rngA:              Ring[A]
  // def binA:              BinaryRepresentation[A]
  // def mul:               (I, W) => M
  // def addVec:            Vec[M] => A
  // def shiftRound:        (A, Int) => A
  // def shiftRoundDynamic: (A, UInt, Bool) => A
  // def actFn:             (A, A) => O
}

class BasicNeuronCompute extends NeuronCompute {
  type I = UInt
  type W = UInt
  type M = UInt
  type A = UInt
  type O = UInt

  def genI = UInt(8.W)
  def genW = UInt(8.W)
  def genM = UInt(16.W)
  def genA = UInt(32.W)
  def genO = UInt(8.W)
}

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