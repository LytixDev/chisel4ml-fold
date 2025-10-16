package empty.hw

import chisel3._
import chisel3.util.{Decoupled, Queue, log2Ceil}
import empty.abstractions.DenseLayer

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
    // TODO: In a folded design, we don't operate on every input in the first cycle.
    //       In fact, we can be much more smart about this. Instead of sending the entire input at the same time we
    //       could instead send it in smaller chunks ensuring that the first chunk contains enough inputs so we can
    //       fully saturate our PEs. This would decrease the amount of wires we need by quite a lot.
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

  // Instead of one huge register array requiring complex dynamic muxing, we organize the inputs into groups that
  // match the access pattern of the "PEs". We access PEsPerOutput inputs at a time sequentially. This matches that
  // access pattern.
  val inputBuffer = Reg(Vec(layer.m, Vec(latency, Vec(layer.PEsPerOutput, nc.genI))))

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

  // Load input buffer and restructre it to fit the organization (essentially grouped by when they are accessed)
  when(firstComputation) {
    for (i <- 0 until layer.m) {
      for (cycle <- 0 until latency) {
        for (pe <- 0 until layer.PEsPerOutput) {
          val flatIdx = cycle * layer.PEsPerOutput + pe
          inputBuffer(i)(cycle)(pe) := io.inputIn.bits(i)(flatIdx)
        }
      }
    }
    computing := true.B
    cycleCounter := 1.U
  }.elsewhen(computing && cycleCounter < (latency - 1).U) {
    cycleCounter := cycleCounter + 1.U
  }.elsewhen(computing && cycleCounter === (latency - 1).U) {
    computing := false.B
    cycleCounter := 0.U
  }

  // Multiply and accumulate
  for (i <- 0 until layer.m) {
    for (j <- 0 until layer.k) {
      // Compute partial sum using layer.PEsPerOutput multipliers
      // NOTE: Despite being (0.U) it works for signed partial sums as well since the bit representation for
      //       0 in unsigned and signed is the same.
      var partialSum = 0.U.asTypeOf(nc.genA)
      for (pe <- 0 until layer.PEsPerOutput) {
        // OPTIMIZATION: Access inputs using static indexing into the restructured buffer.
        // On the first cycle, read from input directly, otherwise from the pre-organized buffer
        val currentCycle = Mux(firstComputation, 0.U, cycleCounter)
        val inputVal = Mux(firstComputation,
          io.inputIn.bits(i)(pe),
          inputBuffer(i)(currentCycle)(pe))

        // Weight indexing.
        // TODO: These are static so perhaps we can do something smart here. BRAM.
        val weightIdx = currentCycle * layer.PEsPerOutput.U + pe.U
        val product = nc.mul(inputVal, weights(weightIdx)(j))
        val productAccum = nc.toAccum(product)
        // NOTE: This creates a huge adder tree
        // TODO: look into "partial product tree for multiplier"
        //       wallace tree?
        // We could also use this current approach when PEsPerOutput is small, but do some pipelining when it is larger
        partialSum = nc.addAccum(partialSum, productAccum)
      }

      // Accumulate
      // NOTE: This first when() is not needed for correctness, but I suppose it reduces power consumption since
      //       it will make the accumulation operation idle when its not computing? Idk.
      when(isComputing) {
        when(firstComputation) {
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
