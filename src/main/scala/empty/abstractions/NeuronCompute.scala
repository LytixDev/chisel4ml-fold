package empty.abstractions

import chisel3._
import empty.hw.QuantizationVariants

// NOTE to self: Sealed trait ensures all variants are defined in this file, enabling exhaustive pattern matching.
// Also: in Chisel we need a concrete hardware instance with width information. That's why the output type also
//       has to be an input


// Stripped down NeuronCompute abstraction from Chisel4ml
// This abstraction allows the same layer logic to work with different numeric types.
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

  // Takes in a single accumulated value for a single value in output matrix and outputs a requantized version
  def requantize(a: A): O

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
  // def requantize(a: A): O = a.asTypeOf(genO)
  def requantize(a: A): O = {
    val rounded = QuantizationVariants.uniformSymmetric(a, -2)
    rounded.asTypeOf(genO)
  }
  def weightScalaToChisel(value: Int): W = value.U.asTypeOf(genW)
}
