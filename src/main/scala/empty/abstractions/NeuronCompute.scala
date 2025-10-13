package empty.abstractions

import chisel3._

/*
 * On Quantization.
 * Q(r) = Int(r/S) - Z
 * NOTE: right now we will constrain it to Z = 0 and S is a power of 2 so it turns into a shift operation.
 * TODO: It would be nice to have a couple of built-in quantization schemes
 *
 * See ShiftRoundable.scala
 */

// NOTE to self: Sealed trait ensures all variants are defined in this file, enabling exhaustive pattern matching.
// Also: in Chisel we need a concrete hardware instance with width information. That's why the output type also
//       has to be an input


// Chisel4ml's ShiftRoundable.scala
object QuantizationUtils {
  // shift is a negative number
  // pAct is the value to be requantized
  def shiftRoundUIntStatic(pAct: UInt, shift: Int): UInt = shift.compare(0) match {
    case 0 => pAct
    case 1 => pAct << shift
    case -1 =>
      if (pAct.getWidth > shift) {
        val shifted = (pAct >> shift.abs).asUInt
        val sign = pAct(pAct.getWidth - 1)
        val nsign = !sign
        val fDec = pAct(shift.abs - 1) // first (most significnat) decimal number
        val rest = if (shift > 1) VecInit(pAct(shift.abs - 2, 0).asBools).reduceTree(_ || _) else true.B
        val carry = (nsign && fDec) || (sign && fDec && rest)
        shifted + carry.asUInt
      } else {
        0.U
      }
  }
}

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
    val rounded = QuantizationUtils.shiftRoundUIntStatic(a, -2)
    rounded.asTypeOf(genO)
  }
  def weightScalaToChisel(value: Int): W = value.U.asTypeOf(genW)
}
