package empty

import empty.abstractions.{BasicNeuronCompute, DenseLayer}
import empty.hw.Pipeline

// TODO: This will be the entrypoint that spins up a server and accepts IR from the frontend.
object Main extends App {
  val useMLIRBackend = false
  println("Creating dummy pipeline for synthesis...")

  val l1m = 1
  val l1n = 784
  val l1k = 32

  val l2m = 1
  val l2n = 32
  val l2k = 10

  val weights1 = Array.fill(l1n, l1k)(scala.util.Random.nextInt(16))
  val weights2 = Array.fill(l2n, l2k)(scala.util.Random.nextInt(16))

  val nc = new BasicNeuronCompute()

  val layer1 = DenseLayer(m = l1m, n = l1n, k = l1k, weights = weights1, PEsPerOutput = 2, neuronCompute = nc)
  val layer2 = DenseLayer(m = l2m, n = l2n, k = l2k, weights = weights2, PEsPerOutput = 1, neuronCompute = nc)

  val layers = Array(layer1, layer2)

  println(s"Pipeline configuration:")
  println(s"  Layer 1: ${layer1.n}x${layer1.k} with ${layer1.PEsPerOutput} PEs (${layer1.n / layer1.PEsPerOutput} cycles)")
  println(s"  Layer 2: ${layer2.n}x${layer2.k} with ${layer2.PEsPerOutput} PEs (${layer2.n / layer2.PEsPerOutput} cycles)")
  println(s"  Total latency: ${layers.map(l => l.n / l.PEsPerOutput).sum} cycles")
  println(s"  Total multipliers: ${layers.map(l => l.m * l.k * l.PEsPerOutput).sum}")

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
    // Classic FIRRTL backend (Chisel 3.x)
    import chisel3.stage.ChiselStage

    (new ChiselStage).emitVerilog(
      new Pipeline(layers),
      Array("--target-dir", "generated")
    )
    // (new ChiselStage).emitFirrtl(
    //   new Pipeline(layers),
    //   Array("--target-dir", "generated")
    // )
    
  }

  println("\nSystemVerilog generated successfully in generated/Pipeline.sv")
}
