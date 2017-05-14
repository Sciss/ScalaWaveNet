package de.sciss.wavenet

import de.sciss.file._
import de.sciss.wavenet.WaveNetModel.Params
import scopt.OptionParser

import scala.collection.immutable.{Seq => ISeq}

/*
  adapted from tensorflow-wavenet, original code
  Copyright (c) 2016 Igor Babuschkin, published under the MIT License (MIT)

 */
object Main {
  final case class Config(
                           corpusDir            : File        = file("corpus"),
                           logDirRoot           : File        = file("logdir"),
                           batchSize            : Int         = 1,
                           checkpointStep       : Int         = 50,
                           numSteps             : Int         = 10000,
                           learningRate         : Double      = 1.0e-3,
                           sampleSize           : Int         = 100000,
                           l2RegStrength        : Double      = 0.0,
                           momentum             : Double      = 0.9,
                           maxCheckpoints       : Int         = 5,
                           modelParams          : Params      = Params()
                         )

  def main(args: Array[String]): Unit = {
    val df = Config()
    val parser = new OptionParser[Config]("ScalaWaveNet") {
      opt[File]('c', "corpus")
        .text(s"The directory containing the corpus of input files. Default: ${df.corpusDir}")
        .action { (v, c) => c.copy(corpusDir = v) }

      opt[File]('o', "log-dir").text(
          "Root directory to place the logging output and generated model. " +
          s"These are stored under a dated subdirectory. Default: ${df.logDirRoot}")
        .action { (v, c) => c.copy(logDirRoot = v) }

      opt[Int]('b', "batch-size")
        .text(s"How many audio files to process at once. Default: ${df.batchSize}")
        .action { (v, c) => c.copy(batchSize = v) }

      opt[Int]('h', "checkpoint-step")
        .text(s"How many steps to save each checkpoint after. Default: ${df.checkpointStep}")
        .action { (v, c) => c.copy(checkpointStep = v) }

      opt[Int]('n', "num-steps")
        .text(s"Number of training steps. Default: ${df.numSteps}")
        .action { (v, c) => c.copy(numSteps = v) }

      opt[Double]('l', "learning-rate")
        .text(s"Learning rate for training. Default: ${df.learningRate}")
        .action { (v, c) => c.copy(learningRate = v) }

      opt[Int]('s', "sample-size")
        .text(s"Concatenate and cut audio files to this many sample frames. Default: ${df.sampleSize}")
        .action { (v, c) => c.copy(sampleSize = v) }

      opt[Double]('2', "l2-reg-strength")
        .text(s"Coefficient in the L2 regularization. Default: ${df.l2RegStrength}")
        .action { (v, c) => c.copy(l2RegStrength = v) }

      opt[Double]('m', "momentum")
        .text(s"Momentum used by sgd or rmsprop optimizer. Ignored by adam optimizer. Default: ${df.momentum}")
        .action { (v, c) => c.copy(momentum = v) }

      opt[Int]('k', "max-checkpoints")
        .text(s"Maximum amount of checkpoints that will be kept alive. Default: ${df.maxCheckpoints}")
        .action { (v, c) => c.copy(maxCheckpoints = v) }

      opt[Int]('r', "sample-rate")
        .text(s"Audio sample rate. Default: ${df.modelParams.sampleRate}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(sampleRate = v)) }

      opt[Seq[Int]]('y', "dilations")
        .text(s"List of dilation taps. Default: ${df.modelParams.dilations.mkString(",")}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(dilations = v.toList)) }

      opt[Int]('f', "filter-width")
        .text(s"Convolution kernel size. Default: ${df.modelParams.filterWidth}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(filterWidth = v)) }

      opt[Boolean]("scalar-input").text(
          "Whether to use the quantized waveform directly as input to the network instead of one-hot encoding it. " +
          s"Default: ${df.modelParams.filterWidth}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(scalarInput = v)) }

      opt[Int]("initial-filter-width").text(
          "The width of the initial filter of the convolution applied to the scalar input. " +
          s"This is only relevant if scalar-input=true. Default: ${df.modelParams.filterWidth}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(initialFilterWidth = v)) }

      opt[Int]('r', "residual-channels")
        .text(s"How many filters to learn for the residual. Default: ${df.modelParams.residualChannels}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(residualChannels = v)) }

      opt[Int]('d', "dilation-channels")
        .text(s"How many filters to learn for the dilated convolution. Default: ${df.modelParams.dilationChannels}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(dilationChannels = v)) }

      opt[Int]('p', "skip-channels").text(
          "How many filters to learn that contribute to the quantized softmax output. " +
          s"Default: ${df.modelParams.skipChannels}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(skipChannels = v)) }

      opt[Int]('q', "skip-channels").text(
          "How many amplitude values to use for audio quantization and the corresponding one-hot encoding. " +
          s"Default: ${df.modelParams.quantizationChannels}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(quantizationChannels = v)) }

      opt[Boolean]("use-biases")
        .text(s"Whether to add a bias layer to each convolution. Default: ${df.modelParams.useBiases}")
        .action { (v, c) => c.copy(modelParams = c.modelParams.copy(useBiases = v)) }
    }
    parser.parse(args, df).fold(sys.exit(1))(run)
  }

  def run(config: Config): Unit = {
    validate_directories(config)

    val histograms  = false  // XXX TODO --- adds command line switch
    val gcChannels  = 0
    val gcEnabled   = false

    val reader = AudioReader(
      corpusDir       = config.corpusDir,
//      coord,
      sampleRate      = config.sampleSize,
      gcEnabled       = gcEnabled,
      receptiveField  = WaveNetModel.calculateReceptiveField(config.modelParams),
      sampleSize      = config.sampleSize
      // silenceThreshold = silenceThreshold
    )

    val audioBatch = reader.dequeue(config.batchSize)

    // Create network.
    val net = WaveNetModel(
      batchSize                   = config.batchSize,
      params                      = config.modelParams,
      histograms                  = histograms,
      globalConditionChannels     = gcChannels,
      globalConditionCardinality  = reader.gcCategoryCardinality
    )

    val loss: Double =
//      net.loss(inputBatch = audioBatch, globalConditionBatch = gcIdBatch,
//      l2RegStrength = config.l2RegStrength)
//    val optimizer = optimizerFactory[args.optimizer](
//      learningRate = config.learningRate, momentum = config.momentum)
//    val trainable = tf.trainable_variables()
//    val optim = optimizer.minimize(loss, varList = trainable)
    ???
  }

  /** Validates directory related arguments. */
  def validate_directories(config: Config): Unit = {
//  require (config.logDir.isDefined ^^ config.logDirRoot.isDefined,
//    "--log-dir and --log-dir-root cannot be specified at the same time.")
//
//    require (config.logDir.isDefined ^^ config.restoreFrom,
//  "--logdir and --restore_from cannot be specified at the same "
//  "time. This is to keep your previous model from unexpected "
//  "overwrites.\n"
//  "Use --logdir_root to specify the root of the directory which "
//  "will be automatically created with current date and time, or use "
//  "only --logdir to just continue the training from the last "
//  "checkpoint.")
  }
}