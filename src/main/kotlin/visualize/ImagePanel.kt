package visualize

import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import java.awt.Dimension
import java.awt.Graphics
import java.awt.image.BufferedImage
import java.awt.image.BufferedImage.TYPE_INT_RGB
import javax.swing.JPanel

class ImagePanel(image: FloatArray, imageShape: ImageShape) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val x = (size.width - bufferedImage.width) / 2
        val y = (size.height - bufferedImage.height) / 2
        graphics.drawImage(bufferedImage, x, y, null)
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

private fun FloatArray.toBufferedImage(imageShape: ImageShape): BufferedImage {
    val result = BufferedImage(imageShape.width!!.toInt(), imageShape.height!!.toInt(), TYPE_INT_RGB)
    val rgbArray = copyOf().also {
    }
    rgbArray.forEachIndexed { index, value -> rgbArray[index] = value * 255f }
    result.raster.setPixels(0, 0, result.width, result.height, rgbArray)
    return result
}