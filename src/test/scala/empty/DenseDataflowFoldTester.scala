package empty

import chisel3._
import chiseltest._
import empty.abstractions.{BasicNeuronCompute, DenseLayer}
import empty.hw.DenseDataflowFold
import empty.sim.DenseDataflowFoldSim
import org.scalatest.flatspec.AnyFlatSpec

class DenseDataflowFoldTester extends AnyFlatSpec with ChiselScalatestTester {

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

    val layer = DenseLayer(m = 2, n = 4, k = 2, weights = weights, PEsPerOutput = 2, neuronCompute = new BasicNeuronCompute)

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

    val layer = DenseLayer(m = 2, n = 4, k = 2, weights = weights, PEsPerOutput = 1, neuronCompute = new BasicNeuronCompute)
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

    val layer = DenseLayer(m = 2, n = 4, k = 2, weights = weights, PEsPerOutput = 4, neuronCompute = new BasicNeuronCompute)
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
