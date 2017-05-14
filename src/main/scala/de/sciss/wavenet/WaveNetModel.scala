package de.sciss.wavenet

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.immutable.{Seq => ISeq}

object WaveNetModel {
  final case class Params(
      sampleRate           : Int         = 16000,
      dilations            : ISeq[Int]   = ISeq.tabulate(10 * 5)(i => 1 << (i % 10)),
      filterWidth          : Int         = 2,
      scalarInput          : Boolean     = false,
      initialFilterWidth   : Int         = 32,
      residualChannels     : Int         = 32,
      dilationChannels     : Int         = 32,
      skipChannels         : Int         = 512,
      quantizationChannels : Int         = 256,
      useBiases            : Boolean     = true
    )

  def calculateReceptiveField(params: Params): Int = ???

  def apply(batchSize: Int, params: Params, histograms: Boolean, globalConditionChannels: Int,
            globalConditionCardinality: Int): WaveNetModel =
    new Impl(batchSize = batchSize, params = params, histograms = histograms,
      globalConditionChannels = globalConditionChannels, globalConditionCardinality = globalConditionCardinality)

  private final class Impl(batchSize: Int, params: Params, histograms: Boolean, globalConditionChannels: Int,
                           globalConditionCardinality: Int)
    extends WaveNetModel {

    private def createNetwork(): Nothing = ???

    /** Returns embedding for global condition.
      *
      * @param  globalCondition Either ID of global condition for
      *        tf.nn.embedding_lookup or actual embedding. The latter is
      *        experimental.
      *
      * @return Embedding or None
      */
    def embedGc(globalCondition: Any): Option[Any] = {
      var embedding = None
      if (globalConditionCardinality != 0) {
        // Only lookup the embedding if the global condition is presented
        // as an integer of mutually-exclusive categories ...
        ??? // val embedding_table = self.variables['embeddings']['gc_embedding']
        // embedding = tf.nn.embedding_lookup(embedding_table, global_condition)
      } else if (false /* globalCondition */) {
        // ... else the global_condition (if any) is already provided
        // as an embedding.

        // In this case, the number of global_embedding channels must be
        // equal to the the last dimension of the global_condition tensor.

//        val gc_batch_rank = len(globalCondition.get_shape())
//        val dims_match = (globalCondition.get_shape()[gc_batch_rank - 1] == self.global_condition_channels)
//        if (!dims_match) {
//          sys.error("Shape of global_condition % does not match global_condition_channels %."
//            .format(globalCondition.get_shape(), globalConditionChannels))
//          embedding = global_condition
//        }
        ???
      }

      embedding.map { e =>
        ??? // tf.reshape(e, [self.batch_size, 1, self.global_condition_channels])
      }
    }

    /** One-hot encodes the waveform amplitudes.
      * This allows the definition of the network as a categorical distribution
      * over a finite set of possible amplitudes.
      */
    def oneHot(inputBatch: INDArray): INDArray = {
      // with tf.name_scope('one_hot_encode'):
      val encoded = ???
//      tf.one_hot(
//        input_batch,
//        depth=self.quantization_channels,
//        dtype=tf.float32)
//      shape = [self.batch_size, -1, self.quantization_channels]
//      encoded = tf.reshape(encoded, shape)

      encoded
    }

    def loss(inputBatch: INDArray, globalConditionBatch: Nothing, l2RegStrength: Double, name: String): Double = {

      // with tf.name_scope(name)

      // We mu-law encode and quantize the input audioform.
      val encodedInput  = Ops.muLawEncode(inputBatch, params.quantizationChannels)
      val gcEmbedding   = embedGc(globalConditionBatch)
      val encoded       = oneHot (encodedInput)

      val networkInput0: INDArray = if (params.scalarInput) {
//        tf.reshape(
//          tf.cast(input_batch, tf.float32),
//          [batchSize, -1, 1]
//        )
        ???
      }
      else encoded

      // Cut off the last sample of network input to preserve causality.
//      val networkInputWidth = tf.shape(networkInput0)[1] - 1
      val networkInput     : INDArray = ??? // tf.slice(networkInput0, [0, 0, 0], [-1, networkInputWidth, -1])

      val rawOutput         = ??? // createNetwork(networkInput, gcEmbedding)

      // with tf.name_scope('loss')

      /*

        tf.slice(input_, begin, size, name=None)

        extracts a slice of size size from a tensor input starting at the location specified by begin.

       */

      // Cut off the samples corresponding to the receptive field
      // for the first predicted sample.
      val targetOutput0: INDArray = ???
//      tf.slice(
//        tf.reshape(
//          encoded,
//          [batchSize, -1, params.quantizationChannels]
//        ), [0, receptiveField, 0], [-1, -1, -1]
//      )
      val targetOutput: INDArray = ??? // tf.reshape(targetOutput0, [-1, params.quantizationChannels])
      val prediction: INDArray = ??? // tf.reshape(rawOutput, [-1, params.quantizationChannels])
      val loss: INDArray = ???
//      tf.nn.softmax_cross_entropy_with_logits(
//        logits = prediction,
//        labels = targetOutput
//      )

      val reducedLoss = loss.meanNumber().doubleValue() // tf.reduce_mean(loss)

  //    tf.summary.scalar('loss', reduced_loss)

      if (l2RegStrength == 0) reducedLoss else {
        // L2 regularization for all trainable parameters
//        val coll = tf.trainable_variables().collect {
//          case v if !v.name.contains("bias") => tf.nn.l2_loss(v)
//        }

        val l2Loss = ??? // tf.add_n([coll])

        // Add the regularization term to the loss
        val totalLoss: Double = ??? // reducedLoss + l2RegStrength * l2Loss

//        tf.summary.scalar('l2_loss', l2Loss)
//        tf.summary.scalar('total_loss', totalLoss)
        totalLoss
      }
    }
  }
}
trait WaveNetModel {
  /** Creates a WaveNet network and returns the autoencoding loss.
    * The variables are all scoped to the given name.
    */
  def loss(inputBatch: INDArray, globalConditionBatch: Nothing, l2RegStrength: Double,
           name: String = "wavenet"): Double
}