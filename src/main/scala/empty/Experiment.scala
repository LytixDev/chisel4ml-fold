package empty

import empty.abstractions.{DenseLayer, Identity, IntegerDataType, ReLU, TensorSpec}
import empty.hw.Pipeline
import empty.{Metrics, PipelineMetrics}

import java.io.PrintWriter
import scala.collection.mutable.ArrayBuffer

/**
 * This generates 20 different Pipeline configurations across two dimensions:
 * - Network size (parameters): 10k, 50k, 100k, 250k, 500k
 * - Fold factor (i.e. multipliers per dot product): 1, ~25%, ~50%, 100%
 *
 * The total number of configuration will be the cross product of the two varying dimensions.
 * For each configuration we elaborate 5 times and keep track of the elaboration time.
 */
object Experiment extends App {

  val OUTPUT_DIR = "generated/experiment"

  case class PipelineConfig(
    name: String,
    targetParams: Int,
    // Note that the actual fold percentage has to divisible by ...
    foldPercentage: Double,
    layers: Array[DenseLayer]
  )

  case class ElaborationResult(
    config: PipelineConfig,
    metrics: PipelineMetrics,
    elaborationTimes: Array[Double],  // 5 runs in seconds
    avgElaborationTime: Double,
    verilogName: String,
    actualParams: Int
  )

  /**
   * Generates a 3-layer neural network with approximately targetParams parameters.
   *
   * Network structure: input -> hidden1 -> hidden2 -> output
   * All layers use:
   * - Weights: signed 4 bits
   * - Biases: signed 8 bits
   * - Activations: unsigned 4 bits
   * - ReLU activation on hidden layers
   */
  def generateNetwork(targetParams: Int, foldPercentage: Double, seed: Int = 42): Array[DenseLayer] = {
    // Calculate layer dimensions based on target parameter count
    // We'll use a scaling factor based on the target
    val scale = math.sqrt(targetParams / 10000.0)

    val n1 = (128 * scale).toInt
    val k1 = (64 * scale).toInt
    val k2 = (32 * scale).toInt
    val k3 = 10

    // Ensure dimensions are reasonable and divisible by common fold factors
    def makeCompatible(dim: Int): Int = {
      val rounded = ((dim / 8) * 8).toInt  // Round to multiple of 8
      math.max(8, rounded)
    }

    val n1_adj = makeCompatible(n1)
    val k1_adj = makeCompatible(k1)
    val k2_adj = makeCompatible(k2)

    // Calculate fold factors for each layer based on percentage
    def calculateFoldFactor(inputDim: Int, percentage: Double): Int = {
      if (percentage >= 1.0) {
        // 100% = fully parallel
        inputDim
      } else if (percentage <= 0.0) {
        // Serial
        1
      } else {
        // Calculate based on percentage
        val target = math.max(1, (inputDim * percentage).toInt)
        // Find largest divisor of inputDim that's <= target
        var fold = 1
        for (i <- 1 to inputDim) {
          if (inputDim % i == 0 && i <= target) {
            fold = i
          }
        }
        fold
      }
    }

    val fold1 = calculateFoldFactor(n1_adj, foldPercentage)
    val fold2 = calculateFoldFactor(k1_adj, foldPercentage)
    val fold3 = calculateFoldFactor(k2_adj, foldPercentage)

    val rand = new scala.util.Random(seed)

    val layer1 = DenseLayer.withRandomWeights(
        m = 1, n = n1_adj, k = k1_adj,
        multipliersPerOutputElement = fold1,
        inputBitWidth = 8,
        weightBitWidth = 4,
        outputBitWidth = 8,
        withBias = true,
        activationFunc = ReLU,
        seed = seed
      )

    val layer2 = DenseLayer.withRandomWeights(
      m = 1, n = k1_adj, k = k2_adj,
      multipliersPerOutputElement = fold2,
      inputBitWidth = 8,
      weightBitWidth = 4,
      outputBitWidth = 8,
      withBias = true,
      activationFunc = ReLU,
      seed = seed + 1
    )

    val layer3 = DenseLayer.withRandomWeights(
      m = 1, n = k2_adj, k = k3,
      multipliersPerOutputElement = fold3,
      inputBitWidth = 8,
      weightBitWidth = 4,
      outputBitWidth = 8,
      withBias = true,
      activationFunc = Identity,
      seed = seed + 2
    )

    Array(layer1, layer2, layer3)
  }

  def countParameters(layers: Array[DenseLayer]): Int = {
    Metrics.calculatePipelineMetrics(layers).totalParameters
  }

  def measureElaborationTime(layers: Array[DenseLayer], outputDir: String): Double = {
    import chisel3.stage.ChiselStage
    val start = System.nanoTime()
    (new ChiselStage).emitVerilog(
      new Pipeline(layers),
      Array("--target-dir", outputDir)
    )
    val end = System.nanoTime()
    (end - start) / 1e9  // To seconds
  }

  def runMultipleElaborations(layers: Array[DenseLayer], runs: Int = 5, verilogOutputName: Option[String] = None): (Array[Double], Double) = {
    val times = (0 until runs).map { i =>
      // On the last run, use the final output directory; otherwise use temp
      val outputDir = if (i == runs - 1 && verilogOutputName.isDefined) {
        s"$OUTPUT_DIR/${verilogOutputName.get}"
      } else {
        s"$OUTPUT_DIR/temp_run_$i"
      }
      measureElaborationTime(layers, outputDir)
    }.toArray

    val avg = times.sum / times.length
    (times, avg)
  }

  def runExperiment(): Unit = {
    val paramCounts = Array(10000, 50000, 100000, 250000, 500000)
    val foldPercentages = Array(
      (0.00, "1"),      // Serial
      (0.25, "25"),
      (0.50, "50"),
      (1.00, "100")     // Fully parallel
    )

    val results = ArrayBuffer[ElaborationResult]()
    var configNum = 0
    val totalConfigs = paramCounts.length * foldPercentages.length

    for (targetParams <- paramCounts) {
      for ((foldPct, foldLabel) <- foldPercentages) {
        configNum += 1

        println(s"\n[$configNum/$totalConfigs] Processing configuration:")
        println(s"  Target params: ${targetParams/1000}k")
        println(s"  Fold factor: ${foldLabel}%")

        val layers = generateNetwork(targetParams, foldPct)
        val actualParams = countParameters(layers)

        val configName = s"Pipeline_${targetParams/1000}k_${foldLabel}pct"
        val config = PipelineConfig(configName, targetParams, foldPct, layers)

        println(s"  Actual params: $actualParams")
        println(s"  Layer dimensions:")
        layers.zipWithIndex.foreach { case (layer, idx) =>
          println(f"    Layer ${idx+1}: ${layer.n}%4d x ${layer.k}%4d (fold=${layer.multipliersPerDotProduct}%4d)")
        }

        println(s"  Calculating metrics...")
        val metrics = Metrics.calculatePipelineMetrics(layers)

        val nRuns = 3

        println(s"  Running elaboration (${nRuns} times, generating Verilog on last run)...")
        val (elaborationTimes, avgTime) = runMultipleElaborations(layers, runs = nRuns, verilogOutputName = Some(configName))
        //val elaborationTimes = Array(1.0, 1.0, 1.0, 1.0, 1.0)
        //val avgTime = 1.0

        elaborationTimes.zipWithIndex.foreach { case (time, idx) =>
          println(f"    Run ${idx+1}: ${time}%.3f seconds")
        }
        println(f"    Average: ${avgTime}%.3f seconds")

        val verilogName = s"${configName}.v"

        val result = ElaborationResult(
          config = config,
          metrics = metrics,
          elaborationTimes = elaborationTimes,
          avgElaborationTime = avgTime,
          verilogName = verilogName,
          actualParams = actualParams
        )

        results += result

        println(s"    Completed: $verilogName")
      }
    }

    generateReport(results.toArray)

    println()
    println("Experiment completed!")
    println(s"Report saved to: $OUTPUT_DIR/report.txt")
  }

  def generateReport(results: Array[ElaborationResult]): Unit = {
    val reportPath = s"$OUTPUT_DIR/report.txt"

    // Create directory if it doesn't exist
    val dir = new java.io.File(OUTPUT_DIR)
    dir.mkdirs()

    val writer = new PrintWriter(reportPath)

    try {
      writer.println("=" * 100)
      writer.println("Multi-Pipeline Elaboration Report")
      writer.println("=" * 100)
      writer.println()

      // Summary table
      writer.println("SUMMARY TABLE")
      writer.println("-" * 100)
      writer.println(f"${"Config"}%-30s ${"Params"}%10s ${"Latency"}%10s ${"Multipliers"}%12s ${"Elab Time"}%12s")
      writer.println("-" * 100)

      for (result <- results) {
        val config = result.config
        val metrics = result.metrics

        writer.println(f"${config.name}%-30s ${result.actualParams}%10d ${metrics.totalLatencyCycles}%10d ${metrics.totalMultipliers.count}%12d ${result.avgElaborationTime}%11.3fs")
      }

      writer.println("-" * 100)
      writer.println()
      writer.println()

      writer.println("DETAILED RESULTS")
      writer.println("=" * 100)
      writer.println()

      for ((result, idx) <- results.zipWithIndex) {
        writer.println(s"[${idx+1}/${results.length}] ${result.config.name}")
        writer.println("-" * 100)
        writer.println()

        // Configuration details
        writer.println("Configuration:")
        writer.println(s"  Target parameters: ${result.config.targetParams}")
        writer.println(s"  Actual parameters: ${result.actualParams}")
        writer.println(s"  Verilog file: ${result.verilogName}")
        writer.println()

        // Pipeline metrics
        writer.println("Pipeline Metrics:")
        val m = result.metrics
        writer.println(s"  Total latency (cycles): ${m.totalLatencyCycles}")
        writer.println(s"  Max layer latency (cycles): ${m.maxLayerLatency}")
        writer.println(s"  Total parameters: ${m.totalParameters}")
        writer.println(s"  Total multipliers: ${m.totalMultipliers.count}")
        writer.println(s"  Total adders: ${m.totalAdders.count}")
        writer.println()

        // Layer-by-layer metrics
        writer.println("Layer-by-layer metrics:")
        for ((layerMetrics, layerIdx) <- m.layerMetrics.zipWithIndex) {
          val layer = result.config.layers(layerIdx)
          writer.println(s"  Layer ${layerIdx + 1}:")
          writer.println(s"    Dimensions: ${layer.n} x ${layer.k}")
          writer.println(s"    dotProductSize: ${layer.n}")
          writer.println(s"    multipliersPerDotProduct: ${layer.multipliersPerDotProduct}")
          writer.println(s"    Latency (cycles): ${layerMetrics.latencyCycles}")
          writer.println(s"    Parameters: ${layerMetrics.totalParameters}")
          writer.println(s"    Multipliers: ${layerMetrics.multipliers.count}")
          writer.println(s"    Adders: ${layerMetrics.adders.count}")
        }
        writer.println()

        // Elaboration times
        writer.println("Elaboration Times:")
        result.elaborationTimes.zipWithIndex.foreach { case (time, runIdx) =>
          writer.println(f"  Run ${runIdx + 1}: ${time}%.3f seconds")
        }
        writer.println(f"  Average: ${result.avgElaborationTime}%.3f seconds")
        writer.println()
        writer.println()
      }

      // Statistical summary
      writer.println("=" * 100)
      writer.println("STATISTICAL SUMMARY")
      writer.println("=" * 100)
      writer.println()

      val avgElabTime = results.map(_.avgElaborationTime).sum / results.length
      val minElabTime = results.map(_.avgElaborationTime).min
      val maxElabTime = results.map(_.avgElaborationTime).max

      writer.println(s"Elaboration time statistics:")
      writer.println(f"  Average: ${avgElabTime}%.3f seconds")
      writer.println(f"  Minimum: ${minElabTime}%.3f seconds")
      writer.println(f"  Maximum: ${maxElabTime}%.3f seconds")
      writer.println()

      val avgLatency = results.map(_.metrics.totalLatencyCycles).sum / results.length
      val avgMultipliers = results.map(_.metrics.totalMultipliers.count).sum / results.length

      writer.println(s"Pipeline statistics:")
      writer.println(f"  Average latency: ${avgLatency}%.1f cycles")
      writer.println(f"  Average multipliers: ${avgMultipliers}%.1f")

    } finally {
      writer.close()
    }
  }

  // Run the experiment
  runExperiment()
}
