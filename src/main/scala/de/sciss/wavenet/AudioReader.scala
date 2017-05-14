package de.sciss.wavenet

import de.sciss.file.File
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

object AudioReader {
  def apply(corpusDir: File, sampleRate: Int, sampleSize: Int, receptiveField: Int, gcEnabled: Boolean): AudioReader =
    new Impl(corpusDir = corpusDir, sampleRate = sampleRate, sampleSize = sampleSize,
      receptiveField = receptiveField, gcEnabled = gcEnabled)

  private final class Impl(corpusDir: File, sampleRate: Int, sampleSize: Int, receptiveField: Int,
                           gcEnabled: Boolean)
    extends AudioReader {

    def gcCategoryCardinality: Int = ???

    def dequeue(numElements: Int): DataSetIterator = {
      // .queue.dequeue_many(num_elements)
      ???
    }
  }
}
trait AudioReader {
  def gcCategoryCardinality: Int

  def dequeue(numElements: Int): DataSetIterator
}