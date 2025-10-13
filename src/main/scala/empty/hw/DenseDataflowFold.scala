package empty.hw

import chisel3._
import chisel3.util.{Decoupled, Queue, log2Ceil}
import empty.DenseLayer

/*
 * Dense layer with folding.
 * Uses NeuronCompute abstraction for parameterizable data types (inputs, weights, accumulators, outputs)
 *
 * Total multipliers = m * k * numberOfPEs
 *
 * Each layer has an internal output FIFO for buffering results enabling pipelining
 */
class DenseDataflowFold(layer: DenseLayer, outFifoDepth: Int = 2) extends Module {
  val nc = layer.neuronCompute
  val io = IO(new Bundle{
    val inputIn = Flipped(Decoupled(Vec(layer.m, Vec(layer.n, nc.genI))))
    val outputOut = Decoupled(Vec(layer.m, Vec(layer.k, nc.genO)))
  })

  require(layer.PEsPerOutput >= 1 && layer.PEsPerOutput <= layer.n)
  require(layer.n % layer.PEsPerOutput == 0)
  val latency = layer.n / layer.PEsPerOutput

  // TODO: Consider storing this in BRAM (and what kind of memory is this synthesized into?)
  val weights = VecInit(layer.weights.toIndexedSeq.map { row =>
    VecInit(row.toIndexedSeq.map { w =>
      nc.weightScalaToChisel(w)
    })
  })

  // Computation state
  val cycleCounter = RegInit(0.U(log2Ceil(latency + 1).W))
  val computing = RegInit(false.B)
  // TODO: So we load from the FIFO and into the regs. Can we just load from the FIFO and avoid storing it in the regs?
  //       If we stick with the Reg solution we need to figure out what kind of memory this is synthesized into.
  val inputReg = Reg(Vec(layer.m, Vec(layer.n, nc.genI)))

  // One accumulator per output element (m*k total)
  // TODO: maybe with the FIFOs we can optimize this? i.e maybe we need less
  val accumulators = Reg(Vec(layer.m, Vec(layer.k, nc.genA)))

  // NOTE: Incurs a 1 cycle latency by default w/o the flow = true param
  // TODO: So the FIFOs give us decoupling between layers. However, if two layers are perfectly in sync, they don't
  //       necessarily need to be decoupled and perhaps we could directly wire them together?
  val outputFifo = Module(new Queue(Vec(layer.m, Vec(layer.k, nc.genO)), outFifoDepth, flow=true))

  // Can accept input when not computing
  io.inputIn.ready := !computing

  // Start computing immeditely as the input fires
  val isComputing = io.inputIn.fire || computing
  val firstComputation = io.inputIn.fire && !computing

  when(firstComputation) {
    inputReg := io.inputIn.bits
    computing := true.B
    cycleCounter := 1.U
  }.elsewhen(computing && cycleCounter < (latency - 1).U) {
    cycleCounter := cycleCounter + 1.U
  }.elsewhen(computing && cycleCounter === (latency - 1).U) {
    computing := false.B
    cycleCounter := 0.U
  }

  // Compute and accumulate
  for (i <- 0 until layer.m) {
    for (j <- 0 until layer.k) {
      // Compute partial sum using layer.PEsPerOutput multipliers
      // NOTE: Despite being (0.U) it works for signed partial sums as well since the bit representation for
      //       0 in unsigned and signed is the same.
      var partialSum = 0.U.asTypeOf(nc.genA)
      for (pe <- 0 until layer.PEsPerOutput) {
        val idx = cycleCounter * layer.PEsPerOutput.U + pe.U
        // For the first cycle we use the input data, otherwise we use the regs
        val inputVal = Mux(io.inputIn.fire, io.inputIn.bits(i)(idx), inputReg(i)(idx))
        val product = nc.mul(inputVal, weights(idx)(j))
        val productAccum = nc.toAccum(product)
        partialSum = nc.addAccum(partialSum, productAccum)
      }

      // Accumulate
      // NOTE: This first when() is not needed for correctness, but I suppose it reduces power consumption since
      //       it will make the accumulation operation idle when its not computing? Idk.
      when(isComputing) {
        when(io.inputIn.fire && !computing) {
          // First cycle
          accumulators(i)(j) := partialSum
        }.otherwise {
          accumulators(i)(j) := nc.addAccum(accumulators(i)(j), partialSum)
        }
      }
    }
  }

  // Connect computation results to output FIFO
  // TODO: Are there scenarios where the downstream layer can start eagerly working on partial results?
  outputFifo.io.enq.valid := RegNext(isComputing && cycleCounter === (latency - 1).U, false.B)
  // TODO: Its not necessary to calculate the requantization in each cycle
  outputFifo.io.enq.bits := VecInit(accumulators.map(row =>
    VecInit(row.map(acc => nc.requantize(acc)))
  ))

  // External output comes from the output FIFO
  io.outputOut <> outputFifo.io.deq
}