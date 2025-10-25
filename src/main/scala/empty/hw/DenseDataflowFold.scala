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
  val nc = empty.abstractions.NeuronCompute(layer)
  val io = IO(new Bundle{
    // TODO: In a folded design, we don't operate on every input in the first cycle.
    //       In fact, we can be much more smart about this. Instead of sending the entire input at the same time we
    //       could instead send it in smaller chunks ensuring that the first chunk contains enough inputs so we can
    //       fully saturate our PEs. This would decrease the amount of wires we need by quite a lot.
    val inputIn = Flipped(Decoupled(Vec(layer.input.rows, Vec(layer.input.cols, nc.genI))))
    val outputOut = Decoupled(Vec(layer.input.rows, Vec(layer.weights.cols, nc.genO)))
  })

  require(layer.PEsPerOutput >= 1 && layer.PEsPerOutput <= layer.input.cols)
  require(layer.input.cols % layer.PEsPerOutput == 0)
  val latency = layer.input.cols / layer.PEsPerOutput

  // TODO: Consider storing this in BRAM (and what kind of memory is this synthesized into?)
  // Organizes the weights into PEs first, then latency/cycle, then the out
  // weights(pe)(cycle)(output_j)
  val weights = VecInit(
    (0 until layer.PEsPerOutput).map { pe =>
      VecInit((0 until latency).map { cycle =>
        VecInit((0 until layer.weights.cols).map { j =>
          val flatIdx = cycle * layer.PEsPerOutput + pe
          nc.weightScalaToChisel(layer.weights.data(flatIdx)(j))
        })
      })
    }
  )

  // Computation state
  val cycleCounter = RegInit(0.U(log2Ceil(latency + 1).W))
  val computing = RegInit(false.B)

  // TODO: So we load from the FIFO and into the regs. Can we just load from the FIFO and avoid storing it in the regs?
  //       If we stick with the Reg solution we need to figure out what kind of memory this is synthesized into.

  // Instead of one huge register array requiring complex dynamic muxing, we organize the inputs to match PE access.
  // PE-first organization: Each PE gets its own small time-indexed buffer.
  // NOTE: It is also possible to do latency-first initialization.
  val inputBuffer = Reg(Vec(layer.input.rows, Vec(layer.PEsPerOutput, Vec(latency, nc.genI))))

  // One accumulator per output element (m*k total)
  // TODO: maybe with the FIFOs we can optimize this? i.e maybe we need less
  val accumulators = Reg(Vec(layer.input.rows, Vec(layer.weights.cols, nc.genA)))

  // NOTE: Incurs a 1 cycle latency by default w/o the flow = true param
  // TODO: So the FIFOs give us decoupling between layers. However, if two layers are perfectly in sync, they don't
  //       necessarily need to be decoupled and perhaps we could directly wire them together?
  val outputFifo = Module(new Queue(Vec(layer.input.rows, Vec(layer.weights.cols, nc.genO)), outFifoDepth, flow=true))

  // Can accept input when not computing
  io.inputIn.ready := !computing

  // Start computing immeditely as the input fires
  val isComputing = io.inputIn.fire || computing
  val firstComputation = io.inputIn.fire && !computing

  // Load input buffer in PE-first organization
  when(firstComputation) {
    for (i <- 0 until layer.input.rows) {
      for (pe <- 0 until layer.PEsPerOutput) {
        for (cycle <- 0 until latency) {
          val flatIdx = cycle * layer.PEsPerOutput + pe
          inputBuffer(i)(pe)(cycle) := io.inputIn.bits(i)(flatIdx)
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
  for (i <- 0 until layer.input.rows) {
    for (j <- 0 until layer.weights.cols) {
      // Compute partial sum using layer.PEsPerOutput multipliers
      // NOTE: Despite being (0.U) it works for signed partial sums as well since the bit representation for
      //       0 in unsigned and signed is the same.
      var partialSum = 0.U.asTypeOf(nc.genA)
      for (pe <- 0 until layer.PEsPerOutput) {
        // PE-first indexing.
        // On the first cycle, read from input directly, otherwise from the PE's time-indexed buffer
        val currentCycle = Mux(firstComputation, 0.U, cycleCounter)
        val inputVal = Mux(firstComputation,
          io.inputIn.bits(i)(pe),
          inputBuffer(i)(pe)(currentCycle))

        // PE-first weight access: PE dimension is compile-time constant, only cycle is dynamic.
        // This eliminates the large n-to-1 mux, replacing it with a small latency-to-1 mux per PE.
        val product = nc.mul(inputVal, weights(pe)(currentCycle)(j))
        val productAccum = nc.toAccum(product)
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
          // TODO: Look into tree reduction or similar
          accumulators(i)(j) := nc.addAccum(accumulators(i)(j), partialSum)
        }
      }
    }
  }

  def activationFunction(acc: nc.A): nc.A = {
    layer.activationFunc match {
      // TODO: Ensure the hardware generated for this actually nothing
      case empty.abstractions.Identity => acc
      case empty.abstractions.ReLU => {
        if (!layer.accDt.isSigned) {
          acc
        } else {
          val zero = 0.S.asTypeOf(nc.genA)
          // Mux(acc < zero, zero, acc)
          // We need to explicitly cast to a SInt() because the Bits representation does not contain signedness info :-(
          Mux(acc.asTypeOf(SInt()) < 0.S, zero, acc.asTypeOf(nc.genA))
        }
      }
      case empty.abstractions.Sigmoid => {
        // TODO:
        acc
      }
    }
  }

  // Connect computation results to output FIFO
  // TODO: Are there scenarios where the downstream layer can start eagerly working on partial results?
  outputFifo.io.enq.valid := RegNext(isComputing && cycleCounter === (latency - 1).U, false.B)

  // TODO: Its not necessary to calculate the requantization in each cycle
  // TODO: Is this too much combinational logic for one cycle?
  // First apply the shift to approximate the real value
  // Then apply the activation function
  // Then requantize into the output domain
  outputFifo.io.enq.bits := VecInit(accumulators.map(row =>
    VecInit(row.map(acc => nc.requantize(activationFunction(nc.approxReal(acc)))))
  ))

  // External output comes from the output FIFO
  io.outputOut <> outputFifo.io.deq
}
