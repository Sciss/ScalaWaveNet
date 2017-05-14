package de.sciss.wavenet

import de.sciss.file._
import de.sciss.synth.io.AudioFile
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

object FromScratch {
  private val LOGGER = LoggerFactory.getLogger(FromScratch.getClass)

  def main(args: Array[String]): Unit = run()

  def readAudioFile(f: File, maxFrames: Int = -1): INDArray = {
    val af = AudioFile.openRead(f)
    try {
      val m = if (maxFrames <= 0) Int.MaxValue else maxFrames
      val n = math.min(m, af.numFrames).toInt
      val b = af.buffer(n)
      af.read(b)
      Nd4j.create(b)
    } finally {
      af.close()
    }
  }

  def run(): Unit = {
    val fIn   = file("/") / "data" / "projects" / "Traumarsenal" / "corpus" / "269171.wav"
    val allIn = readAudioFile(fIn, maxFrames = 48000)
    val numCh = allIn.rows()
    val numFr = allIn.columns()

    // ---- Configure network ----
    val layer0B = new GravesLSTM.Builder()
      .activation(Activation.TANH)
      .nIn(numCh).nOut(numCh * 10)

    val layer0 = layer0B.build()

    val layer1B = new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .activation(Activation.IDENTITY).nIn(numCh * 10).nOut(numCh)

    val layer1 = layer1B.build()

    val confB = new NeuralNetConfiguration.Builder()
      .seed(140)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .learningRate(0.0015)

    val confL = confB.list

    confL
      .layer(0, layer0)
      .layer(1, layer1)

    val conf: MultiLayerConfiguration = confL.build()

    val net = new MultiLayerNetwork(conf)
    net.init()

    net.setListeners(new ScoreIterationListener(20))

    val numFrH = numFr / 2
    val trainArr : INDArray = allIn.get(NDArrayIndex.all(),NDArrayIndex.interval(0     , numFrH))
    val testArr  : INDArray = allIn.get(NDArrayIndex.all(),NDArrayIndex.interval(numFrH, numFrH + numFrH))

    val trainData: DataSet = new DataSet(
      trainArr.get(NDArrayIndex.all(),NDArrayIndex.interval(0, trainArr.columns() - 1)).transpose(),
      trainArr.get(NDArrayIndex.all(),NDArrayIndex.interval(1, trainArr.columns()    )).transpose()
    )
    val testData : DataSet = new DataSet(
      testArr .get(NDArrayIndex.all(),NDArrayIndex.interval(0, testArr .columns() - 1)).transpose(),
      testArr .get(NDArrayIndex.all(),NDArrayIndex.interval(1, testArr .columns()    )).transpose()
    )

    // ---- Train network, evaluating the test set performance at each epoch ----
    val nEpochs = 300
    for (i <- 0 until nEpochs) {
      net.fit(trainData)
      LOGGER.info(s"Epoch $i complete. Time series evaluation:")

      // Run regression evaluation on our single column input
      val evaluation  = new RegressionEvaluation(numCh)
      val features    = testData.getFeatureMatrix

      val labels0     = testData.getLabels
      val labels      = labels0.reshape(labels0.rows(), labels0.columns(), 1) // why?
      val predicted   = net.output(features, false)

      evaluation.evalTimeSeries(labels, predicted)

      // Just do `println` here since the logger will shift the shift the columns of the stats
      println(evaluation.stats)
    }
  }
}