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

  // Basic matrix multiply rules
  require(layer.output.rows == layer.input.rows,
    s"Output rows (${layer.output.rows}) must match input rows (${layer.input.rows})")
  require(layer.output.cols == layer.weights.cols,
    s"Output cols (${layer.output.cols}) must match weight cols (${layer.weights.cols})")

  val io = IO(new Bundle{
    // TODO: In a folded design, we don't operate on every input in the first cycle.
    //       In fact, we can be much more smart about this. Instead of sending the entire input at the same time we
    //       could instead send it in smaller chunks ensuring that the first chunk contains enough inputs so we can
    //       fully saturate our PEs. This would decrease the amount of wires we need by quite a lot.
    val inputIn = Flipped(Decoupled(Vec(layer.input.rows, Vec(layer.input.cols, nc.genI))))
    val outputOut = Decoupled(Vec(layer.output.rows, Vec(layer.output.cols, nc.genO)))
  })

  require(layer.multipliersPerDotProduct >= 1 && layer.multipliersPerDotProduct <= layer.input.cols)
  require(layer.input.cols % layer.multipliersPerDotProduct == 0)
  val totalCyclesNeeded = layer.input.cols / layer.multipliersPerDotProduct

  // TODO: Consider storing this in BRAM (and what kind of memory is this synthesized into?)
  // Converts from row-major indexing to a 3D structure based on output col, PE idx, and cycle
  // Layout: weights(j)(pe)(cycle) where j=output dimension, pe=processing element, cycle=time step
  val weights = VecInit(
    (0 until layer.weights.cols).map { j =>
      VecInit((0 until layer.multipliersPerDotProduct).map { pe =>
        VecInit((0 until totalCyclesNeeded).map { cycle =>
          //val flatIdx = pe * totalCyclesNeeded + cycle
          val flatIdx = cycle * layer.multipliersPerDotProduct + pe
          nc.weightScalaToChisel(layer.weights.data(flatIdx)(j))
        })
      })
    }
  )

  // Bias storage (one value per output column)
  val biases: Option[Vec[nc.A]] = layer.bias.map { biasData =>
    VecInit((0 until layer.weights.cols).map { j =>
      nc.biasScalaToChisel(biasData.data(0)(j))
    })
  }

  // Computation state
  val cycleCounter = RegInit(0.U(log2Ceil(totalCyclesNeeded + 1).W))
  val computing = RegInit(false.B)

  // TODO: So we load from the FIFO and into the regs. Can we just load from the FIFO and avoid storing it in the regs?
  //       If we stick with the Reg solution we need to figure out what kind of memory this is synthesized into.

  // Instead of one huge register array requiring complex dynamic muxing, we organize the inputs to match PE access.
  // PE-first organization: Each PE gets its own small time-indexed buffer.
  // NOTE: It is also possible to do latency-first initialization.
  val inputBuffer = Reg(Vec(layer.input.rows, Vec(layer.multipliersPerDotProduct, Vec(totalCyclesNeeded, nc.genI))))

  // One accumulator per output element (m*k total)
  // TODO: maybe with the FIFOs we can optimize this? i.e maybe we need less
  val accumulators = Reg(Vec(layer.output.rows, Vec(layer.output.cols, nc.genA)))

  // NOTE: Incurs a 1 cycle latency by default w/o the flow = true param
  // TODO: So the FIFOs give us decoupling between layers. However, if two layers are perfectly in sync, they don't
  //       necessarily need to be decoupled and perhaps we could directly wire them together?
  val outputFifo = Module(new Queue(Vec(layer.output.rows, Vec(layer.output.cols, nc.genO)), outFifoDepth, flow=true))

  // Can accept input when not computing
  io.inputIn.ready := !computing

  // Start computing immeditely as the input fires
  val isComputing = io.inputIn.fire || computing
  val firstComputation = io.inputIn.fire && !computing

  // Load input buffer in PE-first organization
  when(firstComputation) {
    for (i <- 0 until layer.input.rows) {
      for (pe <- 0 until layer.multipliersPerDotProduct) {
        for (cycle <- 0 until totalCyclesNeeded) {
          val flatIdx = cycle * layer.multipliersPerDotProduct + pe
          inputBuffer(i)(pe)(cycle) := io.inputIn.bits(i)(flatIdx)
        }
      }
    }
    computing := true.B
    cycleCounter := 1.U
  }.elsewhen(computing && cycleCounter < (totalCyclesNeeded - 1).U) {
    cycleCounter := cycleCounter + 1.U
  }.elsewhen(computing && cycleCounter === (totalCyclesNeeded - 1).U) {
    computing := false.B
    cycleCounter := 0.U
  }

  // Multiply and accumulate
  // Performs a partial dot product in each cycle.
  // After all cycles (cycleCounter = totalCyclesNeeded), each accumulator stores the final dot product result.
  for (i <- 0 until layer.input.rows) {
    for (j <- 0 until layer.weights.cols) {
      var partialSum = 0.U.asTypeOf(nc.genA)
      for (pe <- 0 until layer.multipliersPerDotProduct) {
        // These loops will be unrolled, and i, j and pe become "compile-time" constants, i.e. they require no muxing
        // The only muxing required is for the currentCycle index

        // On the first cycle, read from input directly, otherwise from the PE's time-indexed buffer
        val currentCycle = cycleCounter
        val inputVal = Mux(firstComputation,
          io.inputIn.bits(i)(pe),
          inputBuffer(i)(pe)(currentCycle))

        // Weight access: j (output col) and pe are compile-time constants, only cycle is dynamic.
        val product = nc.mul(inputVal, weights(j)(pe)(currentCycle))
        val productAccum = nc.toAccum(product)
        // TODO: Faster reduction?
        //       In each cycle, the partial sum produces a partial dot product result.
        //       Specifically, the terms of the dot product calculated in each cycle is directly equal PEsPerOutput
        //       We could also use this current approach when PEsPerOutput is small, but do some pipelining when it is larger
        partialSum = nc.addAccum(partialSum, productAccum)
      }

      // Accumulate
      // NOTE: This first when() is not needed for correctness, but I suppose it reduces power consumption since
      //       it will make the accumulation operation idle when its not computing? Idk.
      when(isComputing) {
        when(firstComputation) {
          // First cycle
          // If we have bias we just this cycle to add it to the accumulators
          val partialSumWithBias = biases match {
            case Some(biasVec) => nc.addAccum(partialSum, biasVec(j))
            case None => partialSum
          }
          accumulators(i)(j) := partialSumWithBias
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
  outputFifo.io.enq.valid := RegNext(isComputing && cycleCounter === (totalCyclesNeeded - 1).U, false.B)

  // TODO: Its not necessary to calculate all of this in each cycle
  // TODO: Is this too much combinational logic for one cycle?
  // Apply the shift to approximate the real value
  // Then apply the activation function
  // Then requantize into the output domain
  outputFifo.io.enq.bits := VecInit(accumulators.map(row =>
    VecInit(row.map(acc => nc.requantize(activationFunction(nc.approxReal(acc)))))
  ))

  // External output comes from the output FIFO
  io.outputOut <> outputFifo.io.deq
}
