package empty.sim

import empty.abstractions.{DenseLayer, Identity, ReLU, Sigmoid}

// Scala (and the JVM) does not support specific fixed-width integer math
// This makes it a bit more tricky to emulate
// TODO: We need better helpers for quantization here
//       Fully fledged quantiaztion simulation!


// NOTE: We assume that all weights in the layer an be represented with the bit width of the layer and need no
//       further processing.

class DenseDataflowFoldSim(layer: DenseLayer) {
  def compute(inputs: Array[Array[Int]]): Array[Array[Int]] = {
    require(inputs.length == layer.input.rows, s"Input batch size must be ${layer.input.rows}, got ${inputs.length}")
    require(inputs.forall(_.length == layer.input.cols), s"Input feature size must be ${layer.input.cols}")

    val outputs = Array.ofDim[Int](layer.input.rows, layer.weights.cols)

    for (i <- 0 until layer.input.rows) {
      for (j <- 0 until layer.weights.cols) {
        var accumulator = 0

        for (n <- 0 until layer.input.cols) {
          val input = inputs(i)(n)
          val weight = layer.weights.data(n)(j)

          val product = mul(input, weight)

          // TODO: We need to handle overflow and signedness
          accumulator = accumulator + product
        }

        val shamt = layer.input.shamt + layer.weights.spec.shamt
        val approxReal = applyShift(accumulator, shamt)

        val activated = applyActivation(approxReal)

        val requantized = applyShift(activated, layer.output.shamt)

        //val requantized = quantize(activated, layer.output.dt.bitWidth, layer.output.dt.isSigned)

        outputs(i)(j) = requantized
      }
    }

    outputs
  }

  private def applyShift(value: Int, shamt: Int): Int = {
    return if (shamt == 0) {
      value
    } else if (shamt > 0) {
      value << shamt
    } else {
      value >> -shamt
    }
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
    quantize(product, layer.mulDt.bitWidth, layer.mulDt.isSigned)
  }

  private def applyActivation(value: Int): Int = {
    layer.activation match {
      case Identity => value
      case ReLU => {
        if (value < 0) 0 else value
      }
      case Sigmoid => {
        // TODO
        value
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