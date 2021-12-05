package visualize

import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import utils.IMAGE_SIZE
import utils.NUM_CHANNELS
import utils.REVERSED_LABELS_DICT
import javax.swing.JFrame

fun showImages(dataset: OnFlyImageDataset) {
    val batchIter: Dataset.BatchIterator = dataset.batchIterator(
        8
    )

    for (i in 1..5) {
        val rawImage = batchIter.next().x[2]

        val frame = JFrame("Image")
        frame.contentPane.add(ImagePanel(rawImage, ImageShape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)))
        frame.setSize(1000, 1000)
        frame.pack()
        frame.isVisible = true
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        frame.isResizable = true
    }
}

fun plotSpecialPredictions(model: TensorFlowInferenceModel, dataset: OnFlyImageDataset) {
    for (number in 0..1) {
        val realLabel = dataset.getY(number).toInt()
        val predictedLabel = model.predict(dataset.getX(number))

        val frame =
            JFrame("Real label: ${REVERSED_LABELS_DICT[realLabel]} | Predicted label: ${REVERSED_LABELS_DICT[predictedLabel]}")
        frame.contentPane.add(ImagePanel(dataset.getX(number), ImageShape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)))
        frame.setSize(1000, 1000)
        frame.pack()
        frame.isVisible = true
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        frame.isResizable = true
    }
}