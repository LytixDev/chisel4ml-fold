package empty.sim

import empty.abstractions.DenseLayer

// Scala (and the JVM) does not support specific fixed-width integer math
// This makes it a bit more tricky to emulate
// TODO: We need better helpers for quantization here
//       Fully fledged quantiaztion simulation!


// NOTE: We assume that all weights in the layer an be represented with the bit width of the layer and need no
//       further processing.

class DenseDataflowFoldSim(layer: DenseLayer) {
  private val scheme = layer.quantizationScheme

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

          val product = mul(input, weight)

          // TODO: We need to handle overflow and signedness
          accumulator = accumulator + product
        }

        val requantized = requantize(accumulator)

        outputs(i)(j) = requantized
      }
    }

    outputs
  }

  private def quantize(value: Int, bitWidth: Int, isSigned: Boolean): Int = {
    // TODO: Investiage if this emulates overflow properly for all cases
    val mask = (1 << bitWidth) - 1
    val masked = value & mask

    if (isSigned) {
      // Sign extend if negative
      val signBit = 1 << (bitWidth - 1)
      if ((masked & signBit) != 0) {
        // Negative: extend with ones
        masked | ~mask
      } else {
        masked
      }
    } else {
      masked
    }
  }

  private def mul(input: Int, weight: Int): Int = {
    // TODO: does this work when the inputs and weights have different signedness and bit-widths?
    val product = input * weight
    quantize(product, scheme.mult.bitWidth, scheme.mult.isSigned)
  }

  private def requantize(accum: Int): Int = {
    val shift = -2
    val rounded = QuantizationVariants.uniformSymmetric(accum, shift)
    quantize(rounded, scheme.output.bitWidth, scheme.output.isSigned)
  }

  def printMatrix(name: String, matrix: Array[Array[Int]]): Unit = {
    println(s"$name:")
    matrix.foreach { row =>
      println(row.mkString("[", ", ", "]"))
    }
  }
}
