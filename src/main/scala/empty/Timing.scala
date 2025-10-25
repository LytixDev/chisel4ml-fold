package empty


object Timing {
  private val enabled = true

  def apply[R](label: String)(block: => R): R = {
    if (enabled) {
      val start = System.nanoTime()
      val result = block
      val end = System.nanoTime()
      println(f"  $label took ${(end - start) / 1e9}%.3f seconds")
      result
    } else {
      block
    }
  }
}