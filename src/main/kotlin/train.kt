import data.getCatsDogsDataset
import log.CustomCallback
import model.modelVGG11
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import utils.EPOCHS
import utils.PATH_TO_MODEL
import utils.TEST_BATCH_SIZE
import utils.TRAINING_BATCH_SIZE
import java.io.File

fun main() {
    val (train, test) = getCatsDogsDataset()

    modelVGG11.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY,
            callback = CustomCallback()
        )

        val start = System.currentTimeMillis()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)
        println("Training time: ${(System.currentTimeMillis() - start) / 1000f}")

        it.save(File(PATH_TO_MODEL), writingMode = WritingMode.OVERRIDE)


        val accuracyTest = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]


        println("Accuracy on test data: $accuracyTest")


    }
}

