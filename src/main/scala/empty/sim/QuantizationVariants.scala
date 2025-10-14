package empty.sim

import empty.abstractions.QuantizationOps

object QuantizationVariants extends QuantizationOps[Int] {

  def uniformSymmetric(value: Int, shift: Int): Int = {
    // TODO: This should be parameterizable based on the NeuronCompute configuration
    shiftRoundUIntStatic(value, shift, accumWidth = 32)
  }

  private def shiftRoundUIntStatic(value: Int, shift: Int, accumWidth: Int): Int = {
    shift.compare(0) match {
      case 0 => value
      case 1 =>
        (value << shift) & ((1L << accumWidth) - 1).toInt
      case -1 =>
        val shiftAbs = shift.abs
        if (accumWidth > shiftAbs) {
          val shifted = value >>> shiftAbs
          val sign = (value >>> (accumWidth - 1)) & 1
          val nsign = 1 - sign
          val fDec = (value >>> (shiftAbs - 1)) & 1
          val rest = if (shiftAbs > 1) {
            val mask = (1 << (shiftAbs - 1)) - 1
            (value & mask) != 0
          } else {
            true
          }
          val carry = if ((nsign == 1 && fDec == 1) || (sign == 1 && fDec == 1 && rest)) 1 else 0
          shifted + carry
        } else {
          0
        }
    }
  }
}
