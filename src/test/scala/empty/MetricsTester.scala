package empty

import empty.abstractions.{DenseLayer, IntegerDataType, TensorSpec, TensorData}
import empty.example_nets.DummyNet
import org.scalatest.flatspec.AnyFlatSpec

class MetricsTester extends AnyFlatSpec {

  "Metrics" should "calculate correctly for DenseLayer" in {
    // Create a simple layer for testing
    val inSpec = TensorSpec(
      rows = 1, cols = 8,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val wData = TensorData(
      spec = TensorSpec(
        rows = 8, cols = 4,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 0
      ),
      data = Array.fill(8, 4)(1)
    )
    val outSpec = TensorSpec(
      rows = 1, cols = 4,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val layer = DenseLayer(
      input = inSpec,
      weights = wData,
      output = outSpec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = true),
      accDt = IntegerDataType(bitWidth = 32, isSigned = true),
      multipliersPerOutputElement = 2
    )

    val metrics = Metrics.calculateDenseLayerMetrics(layer)

    // Verify calculations
    // m=1, n=8, k=4, p=2
    // latency = n/p = 8/2 = 4
    assert(metrics.latencyCycles == 4, s"Expected latency 4, got ${metrics.latencyCycles}")

    // multipliers = m*k*p = 1*4*2 = 8
    assert(metrics.multipliers.count == 8, s"Expected 8 multipliers, got ${metrics.multipliers.count}")
    assert(metrics.multipliers.inputBitWidth == 8)
    assert(metrics.multipliers.weightBitWidth == 4)
    assert(metrics.multipliers.outputBitWidth == 16)

    // adders = m*k*p = 1*4*2 = 8
    assert(metrics.adders.count == 8, s"Expected 8 adders, got ${metrics.adders.count}")
    assert(metrics.adders.bitWidth == 32)

    println(metrics)
  }

  "Metrics" should "calculate correctly for Pipeline" in {
    val layers = DummyNet()
    val metrics = Metrics.calculatePipelineMetrics(layers)

    println(metrics)

    // Layer 0: 1x4 @ 4x2, PEs=1 -> latency = 4/1 = 4
    // Layer 1: 1x2 @ 2x1, PEs=2 -> latency = 2/2 = 1
    // Total latency = 4 + 1 = 5
    assert(metrics.totalLatencyCycles == 5, s"Expected total latency 5, got ${metrics.totalLatencyCycles}")
    assert(metrics.maxLayerLatency == 4, s"Expected max latency 4, got ${metrics.maxLayerLatency}")
    assert(metrics.layerMetrics.length == 2)
  }
}
