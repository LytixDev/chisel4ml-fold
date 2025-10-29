package empty

import empty.abstractions.DenseLayer
import empty.hw.Pipeline
import empty.example_nets.{TinyMLP, MnistMLP, DummyNet}

// TODO: This will be the entrypoint that spins up a server and accepts IR from the frontend.
object Main extends App {
  val useMLIRBackend = false
  println("Creating dummy pipeline for synthesis...")

  //val layers = TinyMLP()
  val layers = MnistMLP()
  //val layers = DummyNet()

  val metrics = Metrics.calculatePipelineMetrics(layers)
  println(metrics)

  println(s"\nGenerating ... ")

  if (useMLIRBackend) {
    // New MLIR-based CIRCT backend (requires Chisel 6+)
    // import circt.stage.ChiselStage
    // ChiselStage.emitSystemVerilogFile(
    //   new Pipeline(layers),
    //   args = Array("--target-dir", "generated")
    // )
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

  println("\nOutput successfully generated in generated/Pipeline.(s)v")
}
