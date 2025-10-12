package empty

import chisel3._


// Stripped down NeuronCompute abstraction from Chisel4ml
// This abstraction allows the same layer logic to work with different numeric types.
// - signed and unsigned ints, fixed point?, floating point?
// - custom quantization schemes
abstract class NeuronCompute {
  type I <: Bits // Input
  type W <: Bits // Weight
  type M <: Bits // Multiplication result
  type A <: Bits // Accumulator
  type O <: Bits // Output

  // Generator methods: Chisel needs concrete instances (not just the types) to clone
  // when creating Vecs, Regs, etc. These methods provide those instances.
  def genI: I
  def genW: W
  def genM: M
  def genA: A
  def genO: O

  // Ops
  def mul(i: I, w: W): M
  def toAccum(m: M): A
  def addAccum(a1: A, a2: A): A
  def quantize(a: A): O

  // Helper to convert Scala Int to weight type (handles signed vs unsigned)
  def weightScalaToChisel(value: Int): W

  // def rngA:              Ring[A]
  // def binA:              BinaryRepresentation[A]
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

  def mul(i: I, w: W): M = (i * w).asUInt
  def toAccum(m: M): A = m.asTypeOf(genA)
  def addAccum(a1: A, a2: A): A = a1 + a2
  // TODO: Actual quantization
  def quantize(a: A): O = a.asTypeOf(genO)
  def weightScalaToChisel(value: Int): W = value.U.asTypeOf(genW)
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