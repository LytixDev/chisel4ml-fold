package empty

import empty.abstractions.{DenseLayer, IntegerDataType, ReLU, TensorData, TensorSpec}
import empty.hw.Pipeline

// TODO: This will be the entrypoint that spins up a server and accepts IR from the frontend.
object Main extends App {
  val useMLIRBackend = false
  println("Creating dummy pipeline for synthesis...")


  def MNIST_MLP(): Array[DenseLayer] = {
    // MNIST MLP with random weights
    // l1: in: 1x784, weights: 784 x 32, out: 1x32
    // l2: in: 1x32, weights: 32x10, out: 1x10

    val rand = new scala.util.Random(42)
    val in1 = TensorSpec(
      rows = 1, cols = 784,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val weights1 = TensorData(
      spec = TensorSpec(
        rows = 784, cols = 32,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 0
      ),
      data = Array.fill(784, 32)(rand.nextInt(8) - 4)  // Random weights in [-4, 3]
    )
    val out1 = TensorSpec(
      rows = 1, cols = 32,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )

    val layer1 = DenseLayer(
      input = in1,
      weights = weights1,
      output = out1,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = true),
      accDt = IntegerDataType(bitWidth = 32, isSigned = true),
      activationFunc = ReLU,
      PEsPerOutput = 784  // 56
    )

    val in2 = TensorSpec(
      rows = 1, cols = 32,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val weights2 = TensorData(
      spec = TensorSpec(
        rows = 32, cols = 10,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 0
      ),
      data = Array.fill(32, 10)(rand.nextInt(8) - 4)
    )
    val out2 = TensorSpec(
      rows = 1, cols = 10,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )

    val layer2 = DenseLayer(
      input = in2,
      weights = weights2,
      output = out2,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = true),
      accDt = IntegerDataType(bitWidth = 32, isSigned = true),
      PEsPerOutput = 32  // Fully parallel (no folding)
    )

    Array(layer1, layer2)
  }

  def printPipelineStatistics(layers: Array[DenseLayer]): Unit = {
    println(s"Pipeline configuration:")
    layers.zipWithIndex.foreach { case (layer, idx) =>
      val cycles = layer.input.cols / layer.PEsPerOutput
      println(s"  Layer ${idx + 1}: ${layer.weights.rows}x${layer.weights.cols} with ${layer.PEsPerOutput} PEs ($cycles cycles)")
    }
    println(s"  Total latency: ${layers.map(l => l.input.cols / l.PEsPerOutput).sum} cycles")
    println(s"  Total multipliers: ${layers.map(l => l.input.rows * l.weights.cols * l.PEsPerOutput).sum}")
  }

  val layers = MNIST_MLP()
  printPipelineStatistics(layers)

  println(s"\nGenerating ... ")

  if (useMLIRBackend) {
    // New MLIR-based CIRCT backend (requires Chisel 6+)
    /*
    import circt.stage.ChiselStage
    ChiselStage.emitSystemVerilogFile(
      new Pipeline(layers),
      args = Array("--target-dir", "generated")
    )
    */
  } else {
    // Classic FIRRTL backend (Chisel 3.x) - requires switching to Chisel 3.x in build.sbt
    // Uncomment the commented lines in build.sbt and comment out the Chisel 6 lines to use this
    //throw new UnsupportedOperationException("FIRRTL backend requires Chisel 3.x. Switch dependencies in build.sbt.")
    import chisel3.stage.ChiselStage
    (new ChiselStage).emitVerilog(
      new Pipeline(layers),
      Array("--target-dir", "generated")
    )
  }

  println("\nOutput successfully generated in generated/Pipeline.sv")
}
