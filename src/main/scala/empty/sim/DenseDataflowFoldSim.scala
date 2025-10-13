package empty.sim

import empty.{DenseLayer, QuantizationUtils}

class DenseDataflowFoldSim(layer: DenseLayer) {
  val nc = layer.neuronCompute

  def compute(inputs: Array[Array[Int]]): Array[Array[Int]] = {
    require(inputs.length == layer.m, s"Input batch size must be ${layer.m}, got ${inputs.length}")
    require(inputs.forall(_.length == layer.n), s"Input feature size must be ${layer.n}")

    val outputs = Array.ofDim[Int](layer.m, layer.k)

    for (i <- 0 until layer.m) {
      for (j <- 0 until layer.k) {
        var accumulator = 0

        for (n <- 0 until layer.n) {
          val input = inputs(i)(n)
          val weight = layer.weights(n)(j)

          // Emulate hardware multiplication
          val product = multiplyAsHardware(input, weight)

          accumulator += product
        }

        val requantized = requantizeAsHardware(accumulator)

        outputs(i)(j) = requantized
      }
    }

    outputs
  }

  // TODO: The data types is hardcoded rn, we should use the nc to do this properly
  private def multiplyAsHardware(input: Int, weight: Int): Int = {
    // 8 bit unsigned
    val inputMasked = input & 0xFF
    val weightMasked = weight & 0xFF
    val product = inputMasked * weightMasked
    // 16 bit unsigned
    product & 0xFFFF
  }

  private def requantizeAsHardware(accum: Int): Int = {
    val shift = -2
    val rounded = shiftRoundUIntStatic(accum, shift, accumWidth = 32)
    // truncate
    rounded & 0xFF
  }

  private def shiftRoundUIntStatic(value: Int, shift: Int, accumWidth: Int): Int = {
    shift.compare(0) match {
      case 0  => value
      case 1  => (value << shift) & ((1L << accumWidth) - 1).toInt
      case -1 =>
        val shiftAbs = shift.abs
        if (accumWidth > shiftAbs) {
          val shifted = value >>> shiftAbs

          // Rounding logic from shiftRoundUIntStatic
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

  def printMatrix(name: String, matrix: Array[Array[Int]]): Unit = {
    println(s"$name:")
    matrix.foreach { row =>
      println(row.mkString("[", ", ", "]"))
    }
  }
}
