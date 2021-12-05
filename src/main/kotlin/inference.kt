import data.getCatsDogsDataset
import data.getSpecialImagesDataset
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import utils.IMAGE_SIZE
import utils.NUM_CHANNELS
import utils.PATH_TO_MODEL
import visualize.plotSpecialPredictions
import java.io.File

fun main() {
    val (train, test) = getCatsDogsDataset()
    val specialDataset = getSpecialImagesDataset()

    TensorFlowInferenceModel.load(File(PATH_TO_MODEL)).use {
        it.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        plotSpecialPredictions(
            it,
            specialDataset
        )

        val accuracyTest = it.evaluate(dataset = test, metric = Metrics.ACCURACY)
        val accuracySpecial =
            it.evaluate(dataset = specialDataset, metric = Metrics.ACCURACY)

        println("Accuracy on test data: $accuracyTest")
        println("Accuracy on special data: $accuracySpecial")
    }

}