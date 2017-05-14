package de.sciss.wavenet

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

object Ops {
  /** Quantizes waveform amplitudes. */
  def muLawEncode(audio: INDArray, quantizationChannels: Int): INDArray = {
    // with tf.name_scope('encode')

    val mu            = quantizationChannels - 1

    // Perform mu-law companding transformation (ITU-T, 1988).
    // Minimum operation is here to deal with rare large amplitudes caused
    // by resampling.

    import Transforms._
    import org.nd4s.Implicits._

    val safeAudioAbs  = min(abs(audio), 1.0)
    val magnitude     = log(safeAudioAbs * mu + 1.0) / math.log1p(mu)
    val signal        = sign(audio) * magnitude

    // Quantize signal to the specified number of levels.
    round /* tf.to_int32 */((signal + 1) / 2 * mu + 0.5)
  }
}