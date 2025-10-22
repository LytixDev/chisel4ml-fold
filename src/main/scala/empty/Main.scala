package empty

import empty.abstractions.{DenseLayer, QuantizationParams, QuantizationScheme, QTensor}
import empty.hw.Pipeline

// TODO: This will be the entrypoint that spins up a server and accepts IR from the frontend.
object Main extends App {
  val useMLIRBackend = true
  println("Creating dummy pipeline for synthesis...")

  /*
  val l1q = QuantizationScheme(
    input = QuantizationParams(8, false),
    weight = QuantizationParams(4, true),
    mult = QuantizationParams(16, true),
    accum = QuantizationParams(32, true),
    output = QuantizationParams(4, false),
  )

  val l2q = QuantizationScheme(
    input = QuantizationParams(4, false),
    weight = QuantizationParams(4, true),
    mult = QuantizationParams(8, true),
    accum = QuantizationParams(16, true),
    output = QuantizationParams(4, true),
  )

  val l1m = 1
  val l1n = 784
  val l1k = 32

  val l2m = 1
  val l2n = 32
  val l2k = 10

  val weights1 = Array.fill(l1n, l1k)(scala.util.Random.nextInt(16))
  val weights2 = Array.fill(l2n, l2k)(scala.util.Random.nextInt(16))

  // val nc = new BasicNeuronCompute()

  val layer1 = DenseLayer(m = l1m, n = l1n, k = l1k, weights = weights1, PEsPerOutput = 784, quantizationScheme = l1q)
  val layer2 = DenseLayer(m = l2m, n = l2n, k = l2k, weights = weights2, PEsPerOutput = 32, quantizationScheme = l2q)

  val layers = Array(layer1, layer2)
  */

  val l1q = QuantizationScheme(
    input = QuantizationParams(8, false),
    weight = QuantizationParams(4, true),
    mult = QuantizationParams(8, true),
    accum = QuantizationParams(16, true),
    output = QuantizationParams(4, false),
  )

  val l2q = QuantizationScheme(
    input = QuantizationParams(4, false),
    weight = QuantizationParams(4, true),
    mult = QuantizationParams(8, true),
    accum = QuantizationParams(16, true),
    output = QuantizationParams(4, false),
  )

  val in1 = QTensor(rows = 1, cols = 2, data = Array(Array(0, 0)), hasData = true, shamt = -2)
  val weights1 = QTensor(rows = 2, cols = 2, data = Array(Array(-5, 3), Array(-5, 3)), hasData = true, shamt = 1)

  val in2 = QTensor(rows = 1, cols = 2, data = Array(Array()), hasData = false, shamt = -5)
  val weights2 = QTensor(rows = 2, cols = 1, data = Array(Array(-7), Array(4)), hasData = true, shamt = 0)

  val layer1 = DenseLayer(in1, weights1, 2, l1q)
  val layer2 = DenseLayer(in2, weights2, 2, l2q)
  val layers = Array(layer1, layer2)

  println(s"Pipeline configuration:")
  println(s"  Layer 1: ${layer1.weights.rows}x${layer1.weights.cols} with ${layer1.PEsPerOutput} PEs (${layer1.in.cols / layer1.PEsPerOutput} cycles)")
  println(s"  Layer 2: ${layer1.weights.rows}x${layer1.weights.cols} with ${layer1.PEsPerOutput} PEs (${layer1.in.cols / layer1.PEsPerOutput} cycles)")
  println(s"  Total latency: ${layers.map(l => l.in.cols / l.PEsPerOutput).sum} cycles")
  println(s"  Total multipliers: ${layers.map(l => l.in.rows * l.weights.cols * l.PEsPerOutput).sum}")

  // Generate Verilog
  println(s"\nGenerating SystemVerilog using ${if (useMLIRBackend) "MLIR/CIRCT" else "classic FIRRTL"} backend...")

  if (useMLIRBackend) {
    // New MLIR-based CIRCT backend (requires Chisel 6+)
    import circt.stage.ChiselStage
    ChiselStage.emitSystemVerilogFile(
      new Pipeline(layers),
      args = Array("--target-dir", "generated")
    )
  } else {
    // Classic FIRRTL backend (Chisel 3.x) - requires switching to Chisel 3.x in build.sbt
    // Uncomment the commented lines in build.sbt and comment out the Chisel 6 lines to use this
    throw new UnsupportedOperationException("FIRRTL backend requires Chisel 3.x. Switch dependencies in build.sbt.")
    // import chisel3.stage.ChiselStage
    // (new ChiselStage).emitVerilog(
    //   new Pipeline(layers),
    //   Array("--target-dir", "generated")
    // )
  }

  println("\nSystemVerilog generated successfully in generated/Pipeline.sv")
}
