package xyz.multicolorworld

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.time.Duration
import java.time.LocalDateTime

fun main(args: Array<String>) {

    val batchSizeArray = arrayOf(64, 128, 256)
    val hiddenLayerSizeArray = arrayOf(500, 1000, 1500)
    val numEpochsArray = arrayOf(10, 15, 20)

    for (batchSize in batchSizeArray) {
        for (hiddenLayerSize in hiddenLayerSizeArray) {
            for (numEpochs in numEpochsArray) {
                val start = LocalDateTime.now()
                exam(batchSize, hiddenLayerSize, numEpochs).let {
                    val duration = LocalDateTime.now().let { end ->
                        Duration.between(start, end)
                    }
                    println("$batchSize $hiddenLayerSize $numEpochs $it ${duration.seconds}")
                }
            }
        }
    }
}

fun exam(batchSize: Int, hiddenLayerSize: Int, numEpochs: Int): Double {

    val numRows = 28
    val numColumns = 28
    val outputNum = 10
    val rngSeed = 123

    val train = MnistDataSetIterator(batchSize, true, rngSeed)
    val test = MnistDataSetIterator(batchSize, false, rngSeed)

    val conf = NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, DenseLayer.Builder()
                    .nIn(numRows * numColumns)
                    .nOut(hiddenLayerSize)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(hiddenLayerSize)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .pretrain(false).backprop(true)
            .build()

    val model = MultiLayerNetwork(conf).apply {
        init()
        setListeners(ScoreIterationListener(1))
    }

    for (i in 0..numEpochs - 1) {
        model.fit(train)
    }

    val eval = Evaluation(outputNum)
    for (next in test) {
        model.output(next.featureMatrix).let {
            eval.eval(next.labels, it)
        }
    }
    return eval.f1()
}
