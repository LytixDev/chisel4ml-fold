package empty

import chisel3._
import chiseltest._
import empty.abstractions.{DenseLayer, IntegerDataType, ReLU, TensorData, TensorSpec}
import empty.hw.DenseDataflowFold
import empty.sim.DenseDataflowFoldSim
import org.scalatest.flatspec.AnyFlatSpec

class DenseDataflowFoldTester extends AnyFlatSpec with ChiselScalatestTester {

  def runLayer(layer: DenseLayer, input: Array[Array[Int]]): Array[Array[Int]] = {
    var output: Array[Array[Int]] = null

    // TODO: Is it a better idea to pass the dut as a parameter?
    test(new DenseDataflowFold(layer)) { dut =>
      for (i <- 0 until layer.input.rows) {
        for (j <- 0 until layer.input.cols) {
          val value = if (layer.input.dt.isSigned) {
            input(i)(j).S.asInstanceOf[dut.nc.I]
          } else {
            input(i)(j).U.asInstanceOf[dut.nc.I]
          }
          dut.io.inputIn.bits(i)(j).poke(value)
        }
      }

      dut.io.inputIn.valid.poke(true.B)
      dut.io.outputOut.ready.poke(true.B)
      dut.clock.step(1)
      dut.io.inputIn.valid.poke(false.B)

      // Wait for output to become valid
      while (!dut.io.outputOut.valid.peek().litToBoolean) {
        dut.clock.step(1)
      }

      output = Array.ofDim[Int](layer.output.rows, layer.output.cols)
      for (i <- 0 until layer.output.rows) {
        for (j <- 0 until layer.output.cols) {
          output(i)(j) = dut.io.outputOut.bits(i)(j).peek().litValue.toInt
        }
      }
    }

    output
  }

  "DenseDataflowFold" should "werk for a simple manually computed test" in {
    // I have computed the expected results by hand

    val input = Array(Array(1, 2))
    val weights = Array(
      Array(-1, -8),
      Array(6, 0),
    )
    val expected = Array(Array(44, -32))

    val in = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 4, isSigned = true),
      shamt = 0
    )
    val w = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 2,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 2
      ),
      data = weights
    )
    val outputSpec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = true),
      shamt = 0
    )

    val layer = DenseLayer(
      input = in,
      weights = w,
      output = outputSpec,
      mulDt = IntegerDataType(bitWidth = 8, isSigned = true),
      accDt = IntegerDataType(bitWidth = 8, isSigned = true),
      multipliersPerOutputElement = 2
    )

    val output = runLayer(layer, input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(output(i)(j) == expected(i)(j), s"Hardware output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${output(i)(j)}")
      }
    }

    // We take this as an oppurtonity to test the simulator as well
    val sim = new DenseDataflowFoldSim(layer)
    val simOutput = sim.compute(input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(simOutput(i)(j) == expected(i)(j), s"Simulator output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${simOutput(i)(j)}")
      }
    }
  }

  "DenseDataflowFold" should "werk for a simple manually computed test with ReLU" in {
    // I have computed the expected results by hand

    val input = Array(Array(1, 2))
    val weights = Array(
      Array(-1, -8),
      Array(6, 0),
    )
    val expected = Array(Array(44, 0))

    val in = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 4, isSigned = true),
      shamt = 0
    )
    val w = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 2,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 2
      ),
      data = weights
    )
    val outputSpec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = true),
      shamt = 0
    )

    val layer = DenseLayer(
      input = in,
      weights = w,
      output = outputSpec,
      mulDt = IntegerDataType(bitWidth = 8, isSigned = true),
      accDt = IntegerDataType(bitWidth = 8, isSigned = true),
      activationFunc = ReLU,
      multipliersPerOutputElement = 2
    )

    val output = runLayer(layer, input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(output(i)(j) == expected(i)(j), s"Hardware output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${output(i)(j)}")
      }
    }

    // We take this as an oppurtonity to test the simulator as well
    val sim = new DenseDataflowFoldSim(layer)
    val simOutput = sim.compute(input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(simOutput(i)(j) == expected(i)(j), s"Simulator output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${simOutput(i)(j)}")
      }
    }
  }

  "DenseDataflowFold" should "werk for a simple manually computed test with ReLU and bias" in {
    // I have computed the expected results by hand

    val input = Array(Array(1, 2))
    val weights = Array(
      Array(-1, -8),
      Array(6, 0),
    )
    val biases = Array(Array(2, 3))
    val expected = Array(Array(44 + 8, 0))

    val in = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 4, isSigned = true),
      shamt = 0
    )
    val w = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 2,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 2
      ),
      data = weights
    )
    val b = TensorData(
      spec = TensorSpec(
        rows = 1, cols = 2,
        dt = IntegerDataType(bitWidth = 8, isSigned = true), // NOTE: unused
        shamt = 0 // NOTE: unused
      ),
      data = biases
    )
    val outputSpec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = true),
      shamt = 0
    )

    val layer = DenseLayer(
      input = in,
      weights = w,
      output = outputSpec,
      bias = Some(b),
      mulDt = IntegerDataType(bitWidth = 8, isSigned = true),
      accDt = IntegerDataType(bitWidth = 8, isSigned = true),
      activationFunc = ReLU,
      multipliersPerOutputElement = 2
    )

    val output = runLayer(layer, input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(output(i)(j) == expected(i)(j), s"Hardware output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${output(i)(j)}")
      }
    }

    // We take this as an oppurtonity to test the simulator as well
    val sim = new DenseDataflowFoldSim(layer)
    val simOutput = sim.compute(input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(simOutput(i)(j) == expected(i)(j), s"Simulator output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${simOutput(i)(j)}")
      }
    }
  }

  "DenseDataflowFold" should "werk for a simple manually computed test with ReLU and requantization" in {
    // I have computed the expected results by hand

    val input = Array(Array(1, 2))
    val weights = Array(
      Array(-1, -8),
      Array(6, 0),
    )
    val expected = Array(Array(11, 0))

    val in = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 4, isSigned = true),
      shamt = 0
    )
    val w = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 2,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 2
      ),
      data = weights
    )
    val outputSpec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = true),
      shamt = -2
    )

    val layer = DenseLayer(
      input = in,
      weights = w,
      output = outputSpec,
      mulDt = IntegerDataType(bitWidth = 8, isSigned = true),
      accDt = IntegerDataType(bitWidth = 8, isSigned = true),
      activationFunc = ReLU,
      multipliersPerOutputElement = 2
    )

    val output = runLayer(layer, input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(output(i)(j) == expected(i)(j), s"Hardware output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${output(i)(j)}")
      }
    }

    // We take this as an oppurtonity to test the simulator as well
    val sim = new DenseDataflowFoldSim(layer)
    val simOutput = sim.compute(input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(simOutput(i)(j) == expected(i)(j), s"Simulator output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${simOutput(i)(j)}")
      }
    }
  }

  "DenseDataflowFold" should "werk with different input and weight bit widths" in {
    // I have computed the expected results by hand

    val input = Array(Array(8, -9))
    val weights = Array(
      Array(-1, -8),
      Array(6, 0),
    )
    val expected = Array(Array(-496, -512))

    val in = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = true),
      shamt = 1
    )
    val w = TensorData(
      spec = TensorSpec(
        rows = 2, cols = 2,
        dt = IntegerDataType(bitWidth = 4, isSigned = true),
        shamt = 2
      ),
      data = weights
    )
    val outputSpec = TensorSpec(
      rows = 1, cols = 2,
      dt = IntegerDataType(bitWidth = 16, isSigned = true),
      shamt = 0
    )

    val layer = DenseLayer(
      input = in,
      weights = w,
      output = outputSpec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = true),
      accDt = IntegerDataType(bitWidth = 16, isSigned = true),
      multipliersPerOutputElement = 1 // Should work with 2 as well
    )

    val output = runLayer(layer, input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(output(i)(j) == expected(i)(j), s"Hardware output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${output(i)(j)}")
      }
    }

    val sim = new DenseDataflowFoldSim(layer)
    val simOutput = sim.compute(input)
    for (i <- 0 until expected.length) {
      for (j <- 0 until expected(i).length) {
        assert(simOutput(i)(j) == expected(i)(j), s"Simulator output mismatch at [$i][$j]: expected ${expected(i)(j)}, got ${simOutput(i)(j)}")
      }
    }
  }

  "DenseDataflowFold" should "compute 2x4 matrix multiplication with 2 PEs" in {
    val input = Array(
      Array(1, 2, 3, 4),
      Array(5, 6, 7, 8)
    )
    val weights = Array(
      Array(1, 0),
      Array(0, 1),
      Array(1, 1),
      Array(1, 0)
    )

    val inputSpec = TensorSpec(
      rows = 2, cols = 4,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val weightsData = TensorData(
      spec = TensorSpec(
        rows = 4, cols = 2,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      ),
      data = weights
    )
    val outputSpec = TensorSpec(
      rows = 2, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )

    val layer = DenseLayer(
      input = inputSpec,
      weights = weightsData,
      output = outputSpec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
      accDt = IntegerDataType(bitWidth = 32, isSigned = false),
      multipliersPerOutputElement = 2
    )

    val sim = new DenseDataflowFoldSim(layer)
    val expected = sim.compute(input)

    test(new DenseDataflowFold(layer)) { dut =>
      // Set inputs on the bits field of Decoupled
      for (i <- 0 until 2) {
        for (j <- 0 until 4) {
          dut.io.inputIn.bits(i)(j).poke(input(i)(j).U.asInstanceOf[dut.nc.I])
        }
      }

      // Assert valid and ready to start computation
      dut.io.inputIn.valid.poke(true.B)
      dut.io.outputOut.ready.poke(true.B)
      dut.clock.step(1)
      dut.io.inputIn.valid.poke(false.B)

      // Wait for output to become valid
      while (!dut.io.outputOut.valid.peek().litToBoolean) {
        dut.clock.step(1)
      }

      // Check outputs against simulation
      for (i <- 0 until 2) {
        for (j <- 0 until 2) {
          dut.io.outputOut.bits(i)(j).expect(expected(i)(j).U.asInstanceOf[dut.nc.O])
        }
      }
    }
  }

  "DenseDataflowFold" should "compute 2x4 matrix multiplication with 1 PE" in {
    val input = Array(
      Array(1, 2, 3, 4),
      Array(5, 6, 7, 8)
    )

    val weights = Array(
      Array(1, 0),
      Array(0, 1),
      Array(1, 1),
      Array(1, 0)
    )

    val inputSpec = TensorSpec(
      rows = 2, cols = 4,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val weightsData = TensorData(
      spec = TensorSpec(
        rows = 4, cols = 2,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      ),
      data = weights
    )
    val outputSpec = TensorSpec(
      rows = 2, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )

    val layer = DenseLayer(
      input = inputSpec,
      weights = weightsData,
      output = outputSpec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
      accDt = IntegerDataType(bitWidth = 32, isSigned = false),
      multipliersPerOutputElement = 1
    )
    val sim = new DenseDataflowFoldSim(layer)
    val expected = sim.compute(input)

    test(new DenseDataflowFold(layer)) { dut =>
      for (i <- 0 until 2) {
        for (j <- 0 until 4) {
          dut.io.inputIn.bits(i)(j).poke(input(i)(j).U.asInstanceOf[dut.nc.I])
        }
      }

      // Start computation
      dut.io.inputIn.valid.poke(true.B)
      dut.io.outputOut.ready.poke(true.B)
      dut.clock.step(1)
      dut.io.inputIn.valid.poke(false.B)

      // Wait for output to become valid
      while (!dut.io.outputOut.valid.peek().litToBoolean) {
        dut.clock.step(1)
      }

      for (i <- 0 until 2) {
        for (j <- 0 until 2) {
          dut.io.outputOut.bits(i)(j).expect(expected(i)(j).U.asInstanceOf[dut.nc.O])
        }
      }
    }
  }

  "DenseDataflowFold" should "compute 2x4 matrix multiplication with 4 PEs (no folding)" in {
    // With 4 PEs per output, should complete in 1 cycle (no folding)
    val input = Array(
      Array(1, 2, 3, 4),
      Array(5, 6, 7, 8)
    )

    val weights = Array(
      Array(1, 0),
      Array(0, 1),
      Array(1, 1),
      Array(1, 0)
    )

    val inputSpec = TensorSpec(
      rows = 2, cols = 4,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )
    val weightsData = TensorData(
      spec = TensorSpec(
        rows = 4, cols = 2,
        dt = IntegerDataType(bitWidth = 8, isSigned = false),
        shamt = 0
      ),
      data = weights
    )
    val outputSpec = TensorSpec(
      rows = 2, cols = 2,
      dt = IntegerDataType(bitWidth = 8, isSigned = false),
      shamt = 0
    )

    val layer = DenseLayer(
      input = inputSpec,
      weights = weightsData,
      output = outputSpec,
      mulDt = IntegerDataType(bitWidth = 16, isSigned = false),
      accDt = IntegerDataType(bitWidth = 32, isSigned = false),
      multipliersPerOutputElement = 4
    )
    val sim = new DenseDataflowFoldSim(layer)
    val expected = sim.compute(input)

    test(new DenseDataflowFold(layer)) { dut =>
      for (i <- 0 until 2) {
        for (j <- 0 until 4) {
          dut.io.inputIn.bits(i)(j).poke(input(i)(j).U.asInstanceOf[dut.nc.I])
        }
      }

      // Start computation
      dut.io.inputIn.valid.poke(true.B)
      dut.io.outputOut.ready.poke(true.B)
      dut.clock.step(1)
      dut.io.inputIn.valid.poke(false.B)

      // Wait for output to become valid
      while (!dut.io.outputOut.valid.peek().litToBoolean) {
        dut.clock.step(1)
      }

      for (i <- 0 until 2) {
        for (j <- 0 until 2) {
          dut.io.outputOut.bits(i)(j).expect(expected(i)(j).U.asInstanceOf[dut.nc.O])
        }
      }
    }
  }
}
