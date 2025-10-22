package empty.sim

import empty.abstractions.DenseLayer

class PipelineSim(layers: Array[DenseLayer]) {
  require(layers.nonEmpty, "Pipeline must have at least one layer")
  for (i <- 0 until layers.length - 1) {
    require(layers(i).weights.cols == layers(i + 1).input.cols,
      s"Layer $i output dimension (k=${layers(i).weights.cols}) must match layer ${i+1} input dimension (n=${layers(i+1).input.cols})")
  }

  private val layerSims = layers.map(layer => new DenseDataflowFoldSim(layer))

  def compute(inputs: Array[Array[Int]]): Array[Array[Int]] = {
    layerSims.foldLeft(inputs) { (layerInput, sim) =>
      sim.compute(layerInput)
    }
  }
}