package empty.abstractions

import chisel3._
import empty.hw.QuantizationVariants

// TODO: Later add representation, either Integer or Fixed-point
case class QuantizationParams(bitWidth: Int, isSigned: Boolean)

// input and weight is multiplied into mult
// then the multiplication results (mult) are added into the accumulator
// then the accumulators are requantized into the output
case class QuantizationScheme(
  input: QuantizationParams,
  weight: QuantizationParams,
  mult: QuantizationParams,
  accum: QuantizationParams,
  output: QuantizationParams,
  // TODO: requantization
)

// Old neuroncompute abstraction inherited from chisel4ml
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

// Factory object that creates NeuronCompute implementations from QuantizationScheme
object NeuronCompute {
  def apply(scheme: QuantizationScheme): NeuronCompute = makeNeuronCompute(scheme)

  def makeNeuronCompute(scheme: QuantizationScheme): NeuronCompute = {
  new NeuronCompute {
    private def paramsToChisel(params: QuantizationParams): Bits = {
      if (params.isSigned) SInt(params.bitWidth.W)
      else UInt(params.bitWidth.W)
    }

    type I = Bits
    type W = Bits
    type M = Bits
    type A = Bits
    type O = Bits

    def genI: I = paramsToChisel(scheme.input)
    def genW: W = paramsToChisel(scheme.weight)
    def genM: M = paramsToChisel(scheme.mult)
    def genA: A = paramsToChisel(scheme.accum)
    def genO: O = paramsToChisel(scheme.output)

    def mul(i: I, w: W): M = {
      (scheme.input.isSigned, scheme.weight.isSigned) match {
        case (false, false) => (i.asUInt * w.asUInt).asTypeOf(genM)
        case (false, true)  => (i.asUInt * w.asSInt).asTypeOf(genM)
        case (true, false)  => (i.asSInt * w.asUInt).asTypeOf(genM)
        case (true, true)   => (i.asSInt * w.asSInt).asTypeOf(genM)
      }
    }

    def toAccum(m: M): A = m.asTypeOf(genA)

    def addAccum(a1: A, a2: A): A = {
      if (scheme.accum.isSigned) {
        (a1.asSInt + a2.asSInt).asTypeOf(genA)
      } else {
        (a1.asUInt + a2.asUInt).asTypeOf(genA)
      }
    }

    def requantize(a: A): O = {
      // TODO: Implement proper quantization based on scheme
      val rounded = QuantizationVariants.uniformSymmetric(a.asUInt, -2)
      rounded.asTypeOf(genO)
    }

    def weightScalaToChisel(value: Int): W = {
      if (scheme.weight.isSigned) {
        value.S.asTypeOf(genW)
      } else {
        value.U.asTypeOf(genW)
      }
    }
  }
  }
}

/*
// OLD manual way
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
*/