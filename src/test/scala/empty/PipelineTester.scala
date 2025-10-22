package empty

import chisel3._
import chiseltest._
import chiseltest.simulator.TreadleBackendAnnotation
import empty.abstractions.{DenseLayer, IntegerDataType, TensorSpec, TensorData}
import empty.hw.Pipeline
import empty.sim.PipelineSim
import org.scalatest.flatspec.AnyFlatSpec

import scala.util.Random

class PipelineTester extends AnyFlatSpec with ChiselScalatestTester {

  "Pipeline" should "work with layers taking different amount of cycles" in {
    // Layer 1: 1x4 @ 4x2, with 1 PEs for each output (takes 4 cycles)
    // Layer 2: 1x2 @ 2x1, with 2 PEs for each output (takes 1 cycle)

    val input = Array(Array(1, 2, 3, 4))

    val weights1 = Array(
      Array(1, 0),
      Array(0, 1),
      Array(1, 1),
      Array(0, 1)
    )
    val weights2 = Array(
      Array(2),
      Array(3)
    )

    //val expected = 35;
    val expectedCycles = 5;

    // Layer 1
    val in1Spec = TensorSpec(
      rows = 1, cols = 4,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val w1Data = TensorData(
      spec = TensorSpec(
        rows = 4, cols = 2,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      ),
      data = weights1
    )
    val out1Spec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val layer1 = DenseLayer(
      input = in1Spec,
      weights = w1Data,
      output = out1Spec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
      accDt = IntegerDataType(bitWidth = 32, isSigned = false),
      PEsPerOutput = 1
    )

    // Layer 2
    val in2Spec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val w2Data = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 1,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      ),
      data = weights2
    )
    val out2Spec = TensorSpec(
      rows = 1, cols = 1,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val layer2 = DenseLayer(
      input = in2Spec,
      weights = w2Data,
      output = out2Spec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
      accDt = IntegerDataType(bitWidth = 32, isSigned = false),
      PEsPerOutput = 2
    )

    val layers = Array(layer1, layer2)
    val sim = new PipelineSim(layers)
    val expected = sim.compute(input)

    test(new Pipeline(layers)).withAnnotations(Seq(TreadleBackendAnnotation)) { dut =>
      for (i <- 0 until 1) {
        for (j <- 0 until 4) {
          dut.io.inputIn.bits(i)(j).poke(input(i)(j).U.asInstanceOf[dut.firstNc.I])
        }
      }

      dut.io.inputIn.valid.poke(true.B)
      dut.io.outputOut.ready.poke(true.B)
      dut.clock.step(1)
      dut.io.inputIn.valid.poke(false.B)

      var cycles = 1
      while (!dut.io.outputOut.valid.peek().litToBoolean && cycles < 100) {
        dut.clock.step(1)
        cycles += 1
      }

      //println(s"Computation took $cycles cycles")

      //assert(cycles == expectedCycles, s"Expected $expectedCycles cycles but got $cycles")
      dut.io.outputOut.bits(0)(0).expect(expected(0)(0).U.asInstanceOf[dut.lastNc.O])
    }
  }

  "Pipeline" should "actually work being pipelined" in {
    // 4 layers, all 4x4 @ 4x4, all 1 PE per output (so should take 4 cycles each)
    // The idea is that once L1 is done with the first matmul it will be given a new one immediately afterwards
    // So we will have two inferences in-flight at various stages.
    // This actually tests if the whole pipeline is able to ... actually pipeline!

    val rand = new Random(42)
    var cycles = 0

    // Create 4 identical layers (for simplicity)
    // We use very small numbers for the weights and inputs because the quantization stuff is not implemented yet :-)
    val layers = Array.fill(4) {
      val weights = Array.fill(4, 4)(rand.nextInt(2))
      val inSpec = TensorSpec(
        rows = 1, cols = 4,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      )
      val wData = TensorData(
        spec = TensorSpec(
          rows = 4, cols = 4,
          dt = IntegerDataType(bitWidth = 8, isSigned = false),
          shamt = 0
        ),
        data = weights
      )
      val outSpec = TensorSpec(
        rows = 1, cols = 4,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      )
      DenseLayer(
        input = inSpec,
        weights = wData,
        output = outSpec,
        mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
        accDt = IntegerDataType(bitWidth = 32, isSigned = false),
        PEsPerOutput = 1
      )
    }

    val input1 = Array.fill(1, 4)(rand.nextInt(2))
    val input2 = Array.fill(1, 4)(rand.nextInt(2))

    val pipelineSim = new PipelineSim(layers)
    val expected1 = pipelineSim.compute(input1)
    val expected2 = pipelineSim.compute(input2)

    test(new Pipeline(layers)) { dut =>
      dut.io.outputOut.ready.poke(false.B)

      // Send first input
      for (i <- 0 until 1) {
        for (j <- 0 until 4) {
          dut.io.inputIn.bits(i)(j).poke(input1(i)(j).U.asInstanceOf[dut.firstNc.I])
        }
      }
      dut.io.inputIn.valid.poke(true.B)
      dut.clock.step(1)
      cycles += 1
      dut.io.inputIn.valid.poke(false.B)

      // Wait for first layer to become ready again (FIFO absorbed its output)
      var readyWait = 0
      while (!dut.io.inputIn.ready.peek().litToBoolean && readyWait < 20) {
        dut.clock.step(1)
        cycles += 1
        readyWait += 1
      }
      //println(s"First layer ready again after $readyWait additional cycles")

      // Send second input now that layer is ready
      for (i <- 0 until 1) {
        for (j <- 0 until 4) {
          dut.io.inputIn.bits(i)(j).poke(input2(i)(j).U.asInstanceOf[dut.firstNc.I])
        }
      }
      dut.io.inputIn.valid.poke(true.B)
      dut.clock.step(1)
      cycles += 1
      dut.io.inputIn.valid.poke(false.B)

      // Wait for first output to become valid
      //var cycles = 2
      while (!dut.io.outputOut.valid.peek().litToBoolean && cycles < 100) {
        dut.clock.step(1)
        cycles += 1
      }

      //println(s"First output arrived after $cycles cycles")

      // Capture first output
      val output1 = (0 until 4).map { j =>
        dut.io.outputOut.bits(0)(j).peek().litValue.toInt
      }.toArray
      //println(s"Actual output 1:   ${output1.mkString(", ")}")

      // Verify first output matches expected
      for (j <- 0 until 4) {
        assert(output1(j) == expected1(0)(j),
          s"Output 1 mismatch at index $j: expected ${expected1(0)(j)}, got ${output1(j)}")
      }
      //println("Output 1 matches expected!")

      // Accept first output
      dut.io.outputOut.ready.poke(true.B)
      dut.clock.step(1)

      // Wait for second output
      var cycles2 = 1
      while (!dut.io.outputOut.valid.peek().litToBoolean && cycles2 < 20) {
        dut.clock.step(1)
        cycles2 += 1
      }
      //println(s"Second output arrived $cycles2 cycles after first")

      // Capture second output
      val output2 = (0 until 4).map { j =>
        dut.io.outputOut.bits(0)(j).peek().litValue.toInt
      }.toArray
      //println(s"Actual output 2:   ${output2.mkString(", ")}")

      // Verify second output matches expected
      for (j <- 0 until 4) {
        assert(output2(j) == expected2(0)(j),
          s"Output 2 mismatch at index $j: expected ${expected2(0)(j)}, got ${output2(j)}")
      }
    }
  }


  "Pipeline" should "work for xor example" in {
    val in1Spec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = -1
    )
    val w1Data = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 2,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 1
      ),
      data = Array(Array(-5, 2), Array(-5, 2))
    )
    val out1Spec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 4, isSigned = false),
      shamt = -5
    )

    val layer1 = DenseLayer(
      input = in1Spec,
      weights = w1Data,
      output = out1Spec,
      mulDt = IntegerDataType(bitWidth = 8, isSigned = true),
      accDt = IntegerDataType(bitWidth = 16, isSigned = true),
      PEsPerOutput = 2
    )

    val in2Spec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 4, isSigned = false),
      shamt = -5
    )
    val w2Data = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 1,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 0
      ),
      data = Array(Array(-7), Array(1))
    )
    val out2Spec = TensorSpec(
      rows = 1, cols = 1,
      dt = IntegerDataType(bitWidth = 4, isSigned = false),
      shamt = 0
    )

    val layer2 = DenseLayer(
      input = in2Spec,
      weights = w2Data,
      output = out2Spec,
      mulDt = IntegerDataType(bitWidth = 8, isSigned = true),
      accDt = IntegerDataType(bitWidth = 16, isSigned = true),
      PEsPerOutput = 2
    )

    val layers = Array(layer1, layer2)
    val inputData = Array(Array(0, 0))

    runPipelineTest(layers, Array(inputData))
  }

  // TODO: should also quantization the output once we have quantization in the pipeline
  def matmul(input: Array[Array[Int]], weights: Array[Array[Int]]): Array[Array[Int]] = {
    val m = input.length
    val n = input(0).length
    val k = weights(0).length

    val result = Array.ofDim[Int](m, k)
    for (i <- 0 until m) {
      for (j <- 0 until k) {
        var sum = 0
        for (idx <- 0 until n) {
          sum += input(i)(idx) * weights(idx)(j)
        }
        result(i)(j) = sum
      }
    }
    result
  }

  def generateLayers(nLayers: Int, startN: Int, maxDim: Int, rand: Random): Array[DenseLayer] = {
    var currentN = startN
    Array.tabulate(nLayers) { i =>
      val n = currentN
      val k = rand.nextInt(maxDim - 1) + 2 // k between 2 and maxDim
      // TODO: small weights because quantization is not implemented yet
      val weights = Array.fill(n, k)(rand.nextInt(2))

      // PEsPerOutput must divide n evenly
      val validPEs = (1 to n).filter(pe => n % pe == 0)
      val PEsPerOutput = validPEs(rand.nextInt(validPEs.length))

      currentN = k // Next layer's n must match this layer's k
      val inSpec = TensorSpec(
        rows = 1, cols = n,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      )
      val wData = TensorData(
        spec = TensorSpec(
          rows = n, cols = k,
          dt = IntegerDataType(bitWidth = 8, isSigned = false),
          shamt = 0
        ),
        data = weights
      )
      val outSpec = TensorSpec(
        rows = 1, cols = k,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      )
      DenseLayer(
        input = inSpec,
        weights = wData,
        output = outSpec,
        mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
        accDt = IntegerDataType(bitWidth = 32, isSigned = false),
        PEsPerOutput = PEsPerOutput
      )
    }
  }

  def runPipelineTest(layers: Array[DenseLayer], inputs: Array[Array[Array[Int]]]): Unit = {
    // Calculate expected cycles for each layer
    val layerCycles = layers.map(layer => layer.input.cols / layer.PEsPerOutput)

    // layers.foreach { layer =>
    //   val cycles = layer.input.cols / layer.PEsPerOutput
    //   println(s"Layer n=${layer.input.cols}, PEs=${layer.PEsPerOutput}, cycles=$cycles")
    // }

    val firstLayerCycles = layerCycles(0)
    val totalPipelineCycles = layerCycles.sum
    val delayBetweenPipelinedResults = layerCycles.max
    //val inputInterval = firstLayerCycles  // Send inputs at first layer's rate

    val expectedOutputs = inputs.map { input =>
      var result = input
      for (layer <- layers) {
        result = matmul(result, layer.weights.data)
      }
      result
    }

    test(new Pipeline(layers)) { dut =>
      dut.io.inputIn.valid.poke(false.B)
      dut.io.outputOut.ready.poke(true.B)

      fork {
        for (inferenceIdx <- inputs.indices) {
          // Poke input data
          for (j <- 0 until layers(0).input.cols) {
            dut.io.inputIn.bits(0)(j).poke(inputs(inferenceIdx)(0)(j).U.asInstanceOf[dut.firstNc.I])
          }
          dut.io.inputIn.valid.poke(true.B)
          // println(s"Sending input $inferenceIdx")
          dut.clock.step()
          dut.io.inputIn.valid.poke(false.B)

          for (_ <- 0 until delayBetweenPipelinedResults - 1) {
            dut.clock.step()
          }
        }
        // println("Input thread finished")
      }.fork {
        // This thread checks the output
        for (inferenceIdx <- inputs.indices) {
          var cycles = 0
          // Wait for output to be valid
          while (!dut.io.outputOut.valid.peek().litToBoolean) {
            dut.clock.step()
            cycles += 1
            assert(cycles < 500, s"Timeout waiting for output $inferenceIdx after $cycles cycles")
          }

          val expectedCycles = if (inferenceIdx == 0) totalPipelineCycles else delayBetweenPipelinedResults
          println(s"Output $inferenceIdx arrived after $cycles cycles (expected $expectedCycles)")

          // Verify output
          for (j <- 0 until layers.last.weights.cols) {
            val actual = dut.io.outputOut.bits(0)(j).peek().litValue.toInt
            val expected = expectedOutputs(inferenceIdx)(0)(j)
            assert(actual == expected,
              s"Inference $inferenceIdx, output[$j]: expected $expected, got $actual")
          }

          dut.clock.step()
        }
      }.join()
    }
  }

  /*
  "Pipeline" should "should work for a variety of configs" in {
    val rand = new Random(42)

    for (testNum <- 0 until 10) {
      val nLayers = rand.nextInt(5) + 1

      // Build layers with compatible dimensions
      var currentN = rand.nextInt(6) + 2
      val layers = Array.tabulate(nLayers) { i =>
        val n = currentN
        val k = rand.nextInt(6) + 2
        val weights = Array.fill(n, k)(rand.nextInt(2))

        // PEsPerOutput must divide n evenly
        val validPEs = (1 to n).filter(pe => n % pe == 0)
        val PEsPerOutput = validPEs(rand.nextInt(validPEs.length))

        currentN = k // Next layer's n must match this layer's k
        DenseLayer(m = 1, n = n, k = k, weights = weights, PEsPerOutput = PEsPerOutput, neuronCompute = new BasicNeuronCompute)
      }

      // println(s"\n=== Test $testNum: ${nLayers} layers, dimensions: ${layers.map(l => s"${l.n}x${l.k}(PE=${l.PEsPerOutput})").mkString(" -> ")} ===")

      val numInferences = 2
      val inputs = Array.fill(numInferences) {
        Array.fill(1, layers(0).n)(rand.nextInt(2))
      }

      // Run test with cycle tracking
      runPipelineTest(layers, inputs)
    }
  }
   */

}