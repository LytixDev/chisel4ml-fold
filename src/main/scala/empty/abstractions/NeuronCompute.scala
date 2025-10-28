package empty.abstractions

import chisel3._

// input and weight is multiplied into mult
// then the multiplication results (mult) are added into the accumulator
// then the accumulators are requantized into the output

// Stripped down NeuronCompute abstraction from Chisel4ml
abstract class NeuronCompute {
  type I <: Bits // Input
  type W <: Bits // Weight
  type M <: Bits // Multiplication result
  type A <: Bits // Accumulator
  type O <: Bits // Output

  def genI: I
  def genW: W
  def genM: M
  def genA: A
  def genO: O

  // Ops
  def mul(i: I, w: W): M
  def toAccum(m: M): A
  def addAccum(a1: A, a2: A): A
  //def applyShift(a: A): A

  def approxReal(a: A): A
  def requantize(a: A): O

  // Helper to convert Scala Int to weight type (handles signed vs unsigned)
  def weightScalaToChisel(value: Int): W

  // Helper to convert bias value to accumulator type
  // TODO: We should look into having the bias have its own Bits type and not always rely on the accumulator
  def biasScalaToChisel(value: Int): A
}

// Factory object that creates NeuronCompute implementations from DenseLayers
object NeuronCompute {
  def apply(denseLayer: DenseLayer): NeuronCompute = makeNeuronCompute(denseLayer)

  def makeNeuronCompute(denseLayer: DenseLayer): NeuronCompute = {
  new NeuronCompute {
    private def dtToChisel(dt: IntegerDataType): Bits = {
      if (dt.isSigned) SInt(dt.bitWidth.W)
      else UInt(dt.bitWidth.W)
    }

    type I = Bits
    type W = Bits
    type M = Bits
    type A = Bits
    type O = Bits

    def genI: I = dtToChisel(denseLayer.input.dt)

    def genW: W = dtToChisel(denseLayer.weights.dt)

    def genM: M = dtToChisel(denseLayer.mulDt)

    def genA: A = dtToChisel(denseLayer.accDt)

    def genO: O = dtToChisel(denseLayer.output.dt)

    // This is much faster than the option below, but is it correct?
    // Will i.asSInt reinterpret the bytes as signed? If so we may have a problem here.
    private val mulFunc: (I, W) => M =
      if (denseLayer.mulDt.isSigned) {
        (i: I, w: W) => (i.asSInt * w.asSInt).asTypeOf(genM)
      } else {
        (i: I, w: W) => (i.asUInt * w.asUInt).asTypeOf(genM)
      }

    def mul(i: I, w: W): M = mulFunc(i, w)

    // def mul(i: I, w: W): M = {
    // (denseLayer.input.dt.isSigned, denseLayer.weights.dt.isSigned) match {
    //   case (false, false) => (i.asUInt * w.asUInt).asTypeOf(genM)
    //   case (false, true)  => (i.asUInt * w.asSInt).asTypeOf(genM)
    //   case (true, false)  => (i.asSInt * w.asUInt).asTypeOf(genM)
    //   case (true, true)   => (i.asSInt * w.asSInt).asTypeOf(genM)
    // }
    //}

    def toAccum(m: M): A = m.asTypeOf(genA)

    def addAccum(a1: A, a2: A): A = {
      if (denseLayer.accDt.isSigned) {
        (a1.asSInt + a2.asSInt).asTypeOf(genA)
      } else {
        (a1.asUInt + a2.asUInt).asTypeOf(genA)
      }
    }

    def applyShift(a: A, shamt: Int): A = {
      if (shamt == 0) {
        a.asTypeOf(genO)
      } else if (shamt > 0) {
        (a << shamt).asTypeOf(genO)
      } else {
        (a >> -shamt).asTypeOf(genO)
      }
    }

    def approxReal(a: A): A = {
      val shamt = denseLayer.input.shamt + denseLayer.weights.spec.shamt
      applyShift(a, shamt)
    }

    def requantize(a: A): O = {
      // TODO: do we need to think about rounding?
      applyShift(a, denseLayer.output.shamt)
    }

    def weightScalaToChisel(value: Int): W = {
      if (denseLayer.weights.dt.isSigned) {
        value.S.asTypeOf(genW)
      } else {
        value.U.asTypeOf(genW)
      }
    }

    def biasScalaToChisel(value: Int): A = {
      if (denseLayer.accDt.isSigned) {
        value.S.asTypeOf(genA)
      } else {
        value.U.asTypeOf(genA)
      }
    }
  }
  }
}