package data

import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.dogsCatsDatasetPath
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.FromFolders
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import utils.IMAGE_SIZE
import utils.LABELS_DICT
import utils.NUM_CHANNELS
import utils.TRAIN_TEST_SPLIT_RATIO
import visualize.showImages
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths



fun getCatsDogsDataset(): Pair<OnFlyImageDataset, OnFlyImageDataset> {

    val dogsCatsImages = if (Files.exists(Paths.get("cache/datasets/dogs-vs-cats/"))) {
        "cache/datasets/dogs-vs-cats/"
    } else dogsCatsDatasetPath()

    val preprocessingCatsDogs: Preprocessing = preprocessingPipeline(dogsCatsImages)

    val dataset = OnFlyImageDataset.create(preprocessingCatsDogs).shuffle()

    showImages(dataset)
    return dataset.split(TRAIN_TEST_SPLIT_RATIO)
}

fun getSpecialImagesDataset(): OnFlyImageDataset {
    val specialImages = "cache/datasets/special_images/"

    val preprocessingSpecialImages: Preprocessing = preprocessingPipeline(specialImages)

    val datasetSpecial = OnFlyImageDataset.create(preprocessingSpecialImages)

    return datasetSpecial
}


private fun preprocessingPipeline(dogsCatsImages: String): Preprocessing {
    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = File(dogsCatsImages)
            imageShape = ImageShape(channels = NUM_CHANNELS)
            labelGenerator = FromFolders(mapping = LABELS_DICT)
        }
        transformImage {
            resize {
                outputHeight = IMAGE_SIZE.toInt()
                outputWidth = IMAGE_SIZE.toInt()
                interpolation = InterpolationType.NEAREST
            }

        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }
    return preprocessing
}