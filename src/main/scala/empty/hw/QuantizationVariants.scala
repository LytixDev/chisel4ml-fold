package empty.hw

import chisel3._
import empty.abstractions.QuantizationOps

object QuantizationVariants extends QuantizationOps[UInt] {

  def uniformSymmetric(value: UInt, shift: Int): UInt = {
    shiftRoundUIntStatic(value, shift)
  }

  // Implementation copied from Chisel4ml's ShiftRoundable.scala
  private def shiftRoundUIntStatic(pAct: UInt, shift: Int): UInt = shift.compare(0) match {
    case 0 => pAct
    case 1 => pAct << shift
    case -1 =>
      if (pAct.getWidth > shift) {
        val shifted = (pAct >> shift.abs).asUInt
        val sign = pAct(pAct.getWidth - 1)
        val nsign = !sign
        val fDec = pAct(shift.abs - 1) // first (most significant) decimal number
        val rest = if (shift > 1) VecInit(pAct(shift.abs - 2, 0).asBools).reduceTree(_ || _) else true.B
        val carry = (nsign && fDec) || (sign && fDec && rest)
        shifted + carry.asUInt
      } else {
        0.U
      }
  }
}
