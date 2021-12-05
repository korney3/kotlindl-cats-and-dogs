package utils

const val EPOCHS = 10
const val TRAINING_BATCH_SIZE = 32
const val NUM_CHANNELS = 3L
const val IMAGE_SIZE = 224L
const val SEED = 12L
const val TEST_BATCH_SIZE = 32
const val TRAIN_TEST_SPLIT_RATIO = 0.8
const val LOG_EACH_N_BATCHS = 5
const val PATH_TO_MODEL = "savedmodels/vgg11"
const val NUM_LABELS = 2

val LABELS_DICT = mapOf("cat" to 0, "dog" to 1)
val REVERSED_LABELS_DICT = mapOf(0 to "cat", 1 to "dog")