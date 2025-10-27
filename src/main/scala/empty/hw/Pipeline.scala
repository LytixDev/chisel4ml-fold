package empty.hw

import chisel3._
import chisel3.util.Decoupled
import empty.abstractions.DenseLayer

class Pipeline(layers: Array[DenseLayer]) extends Module {
  require(layers.nonEmpty, "Pipeline must have at least one layer")

  // Validate that layers are compatible (output of layer i matches input of layer i+1)
  // NOTE: This should probably be done by the frontend
  for (i <- 0 until layers.length - 1) {
    require(layers(i).output.cols == layers(i + 1).input.cols,
      s"Layer $i output dimension (k=${layers(i).output.cols}) must match layer ${i+1} input dimension (n=${layers(i+1).input.cols})")
  }

  val firstLayer = layers.head
  val lastLayer = layers.last

  val firstNc = empty.abstractions.NeuronCompute(firstLayer)
  val lastNc = empty.abstractions.NeuronCompute(lastLayer)

  val io = IO(new Bundle {
    val inputIn = Flipped(Decoupled(Vec(firstLayer.input.rows, Vec(firstLayer.input.cols, firstNc.genI))))
    val outputOut = Decoupled(Vec(lastLayer.output.rows, Vec(lastLayer.output.cols, lastNc.genO)))
  })

  val denseModules = layers.map(layer => Module(new DenseDataflowFold(layer)))

  denseModules.head.io.inputIn <> io.inputIn
  for (i <- 0 until denseModules.length - 1) {
    denseModules(i + 1).io.inputIn <> denseModules(i).io.outputOut
  }
  io.outputOut <> denseModules.last.io.outputOut
}
