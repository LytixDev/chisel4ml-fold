package empty

import empty.abstractions.DenseLayer

case class MultiplierSpec(
  count: Int,
  inputBitWidth: Int,
  weightBitWidth: Int,
  outputBitWidth: Int
)

case class AdderSpec(
  count: Int,
  bitWidth: Int
)

case class DenseLayerMetrics(
  latencyCycles: Int,
  multipliers: MultiplierSpec,
  adders: AdderSpec,
  totalParameters: Int
) {
  override def toString: String = {
    s"""DenseLayer:
       |  Latency: $latencyCycles cycles
       |  Multipliers: ${multipliers.count} (${multipliers.inputBitWidth}b x ${multipliers.weightBitWidth}b -> ${multipliers.outputBitWidth}b)
       |  Adders: ${adders.count} (${adders.bitWidth}b)
       |  Total Parameters: $totalParameters
       |""".stripMargin
  }
}

case class PipelineMetrics(
  totalLatencyCycles: Int,
  maxLayerLatency: Int,
  totalMultipliers: MultiplierSpec,
  totalAdders: AdderSpec,
  totalParameters: Int,
  layerMetrics: Seq[DenseLayerMetrics]
) {
  override def toString: String = {
    val layerDetails = layerMetrics.zipWithIndex.map { case (lm, idx) =>
      s"  Layer $idx: ${lm.latencyCycles} cycles, ${lm.multipliers.count} muls, ${lm.adders.count} adds, ${lm.totalParameters} params"
    }.mkString("\n")

    s"""Pipeline Metrics:
       |  Total Latency: $totalLatencyCycles cycles
       |  Max Layer Latency: $maxLayerLatency cycles
       |  Total Multipliers: ${totalMultipliers.count}
       |  Total Adders: ${totalAdders.count}
       |  Total Parameters: $totalParameters
       |
       |Layer Details:
       |$layerDetails
       |""".stripMargin
  }
}

object Metrics {
  def calculateDenseLayerMetrics(layer: DenseLayer): DenseLayerMetrics = {
    val m = layer.input.rows
    val n = layer.input.cols
    val k = layer.weights.cols
    val p = layer.multipliersPerDotProduct

    val latency = n / p

    val multiplierCount = m * k * p
    val multiplierSpec = MultiplierSpec(
      count = multiplierCount,
      inputBitWidth = layer.input.dt.bitWidth,
      weightBitWidth = layer.weights.dt.bitWidth,
      outputBitWidth = layer.mulDt.bitWidth
    )

    // For each output element (m*k total), we need:
    // - (PEsPerOutput - 1) adders to sum the partial products per cycle
    // - 1 adder to accumulate across cycles
    // - ?
    val adderCount = m * k * p
    val adderSpec = AdderSpec(
      count = adderCount,
      bitWidth = layer.accDt.bitWidth
    )

    // Calculate total parameters (weights + biases)
    val weightParams = n * k
    val biasParams = layer.bias.map(_ => k).getOrElse(0)
    val totalParams = weightParams + biasParams

    DenseLayerMetrics(
      latencyCycles = latency,
      multipliers = multiplierSpec,
      adders = adderSpec,
      totalParameters = totalParams
    )
  }

  def calculatePipelineMetrics(layers: Array[DenseLayer]): PipelineMetrics = {
    val layerMetrics = layers.map(calculateDenseLayerMetrics)

    val totalLatency = layerMetrics.map(_.latencyCycles).sum

    // i.e. throughput, or how often we can expect an output going at max speed
    val maxLatency = layerMetrics.map(_.latencyCycles).max

    // NOTE: This is a simplified view since bit-widths may differ per layer
    val totalMultiplierCount = layerMetrics.map(_.multipliers.count).sum
    val totalMultiplierSpec = MultiplierSpec(
      count = totalMultiplierCount,
      inputBitWidth = 0, // Mixed bit-widths
      weightBitWidth = 0,
      outputBitWidth = 0
    )

    val totalAdderCount = layerMetrics.map(_.adders.count).sum
    val totalAdderSpec = AdderSpec(
      count = totalAdderCount,
      bitWidth = 0
    )

    val totalParams = layerMetrics.map(_.totalParameters).sum

    PipelineMetrics(
      totalLatencyCycles = totalLatency,
      maxLayerLatency = maxLatency,
      totalMultipliers = totalMultiplierSpec,
      totalAdders = totalAdderSpec,
      totalParameters = totalParams,
      layerMetrics = layerMetrics
    )
  }
}