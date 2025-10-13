package empty

// TODO: Experiment with the new mlir based backend
import chisel3.stage.ChiselStage
import empty.abstractions.{BasicNeuronCompute, DenseLayer}
import empty.hw.Pipeline

// TODO: This will be the entrypoint that spins up a server and accepts IR from the frontend.
object Main extends App {
  println("Creating dummy pipeline for synthesis...")

  // Create a simple studpid 3-layer network
  // Layer 1: 1x8
  // Layer 2: 1x16
  // Layer 3: 1x8

  val weights1 = Array.fill(8, 16)(scala.util.Random.nextInt(16))
  val weights2 = Array.fill(16, 8)(scala.util.Random.nextInt(16))
  val weights3 = Array.fill(8, 4)(scala.util.Random.nextInt(16))

  val nc = new BasicNeuronCompute()

  val layer1 = DenseLayer(m = 1, n = 8,  k = 16, weights = weights1, PEsPerOutput = 4, neuronCompute = nc)
  val layer2 = DenseLayer(m = 1, n = 16, k = 8,  weights = weights2, PEsPerOutput = 4, neuronCompute = nc)
  val layer3 = DenseLayer(m = 1, n = 8,  k = 4,  weights = weights3, PEsPerOutput = 2, neuronCompute = nc)

  val layers = Array(layer1, layer2, layer3)

  // println(s"Pipeline configuration:")
  // println(s"  Layer 1: ${layer1.n}x${layer1.k} with ${layer1.PEsPerOutput} PEs (${layer1.n / layer1.PEsPerOutput} cycles)")
  // println(s"  Layer 2: ${layer2.n}x${layer2.k} with ${layer2.PEsPerOutput} PEs (${layer2.n / layer2.PEsPerOutput} cycles)")
  // println(s"  Layer 3: ${layer3.n}x${layer3.k} with ${layer3.PEsPerOutput} PEs (${layer3.n / layer3.PEsPerOutput} cycles)")
  // println(s"  Total latency: ${layers.map(l => l.n / l.PEsPerOutput).sum} cycles")
  // println(s"  Total multipliers: ${layers.map(l => l.m * l.k * l.PEsPerOutput).sum}")

  // Generate Verilog
  println("\nGenerating Verilog...")
  (new ChiselStage).emitVerilog(
    new Pipeline(layers),
    Array("--target-dir", "generated")
  )

  println("\nVerilog generated successfully in generated/Pipeline.v")
}
