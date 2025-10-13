package empty.abstractions

/**
 * Quantization interface
 *
 * Essentially the frontend should map the chosen quantization to one of the variants here.
 * Right now we only support uniform symmetric quantization for unsigned ints.
 */
trait QuantizationOps[T] {
  def uniformSymmetric(value: T, shift: Int): T
}