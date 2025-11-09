/*
package empty

object WeightReshapeDemo {
  def main(args: Array[String]): Unit = {
    // Create 4x4 weight matrix
    val weightsData = Array(
      Array("a", "b", "c", "d"),
      Array("e", "f", "g", "h"),
      Array("i", "j", "k", "l"),
      Array("m", "n", "o", "p")
    )

    // Parameters
    val weightsCols = 4  // Number of output columns
    val multipliersPerDotProduct = 2
    val inputCols = 4
    val totalCyclesNeeded = inputCols / multipliersPerDotProduct  // = 2

    println("Original weight matrix (row-major):")
    weightsData.foreach { row =>
      println(row.mkString("[", ", ", "]"))
    }
    println()

    // Reshape into weights(j)(pe)(cycle)
    val weights = Array.ofDim[String](weightsCols, multipliersPerDotProduct, totalCyclesNeeded)

    for (j <- 0 until weightsCols) {
      for (pe <- 0 until multipliersPerDotProduct) {
        for (cycle <- 0 until totalCyclesNeeded) {
          val flatIdx = cycle * multipliersPerDotProduct + pe
          weights(j)(pe)(cycle) = weightsData(flatIdx)(j)
        }
      }
    }

    // Print the reshaped weights
    println("Reshaped weights(j)(pe)(cycle):")
    println()

    for (j <- 0 until weightsCols) {
      println(s"Output column j=$j:")
      for (pe <- 0 until multipliersPerDotProduct) {
        print(s"  PE $pe: [")
        for (cycle <- 0 until totalCyclesNeeded) {
          print(s"${weights(j)(pe)(cycle)}")
          if (cycle < totalCyclesNeeded - 1) print(", ")
        }
        println("]")
      }
      println()
    }

    // Show which original matrix elements each PE handles
    println("Mapping from original matrix:")
    for (j <- 0 until weightsCols) {
      println(s"\nOutput j=$j (column $j of original matrix):")
      for (pe <- 0 until multipliersPerDotProduct) {
        print(s"  PE $pe computes with rows: ")
        val rows = (0 until totalCyclesNeeded).map { cycle =>
          val flatIdx = cycle * multipliersPerDotProduct + pe
          flatIdx
        }
        println(rows.mkString("[", ", ", "]"))
      }
    }
  }
}

 */