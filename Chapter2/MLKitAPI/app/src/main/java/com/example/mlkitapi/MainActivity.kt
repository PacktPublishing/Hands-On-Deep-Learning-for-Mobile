package com.example.mlkitapi


import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import android.widget.Toast.makeText
import com.google.firebase.ml.naturallanguage.FirebaseNaturalLanguage
import com.google.firebase.ml.naturallanguage.languageid.FirebaseLanguageIdentificationOptions
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.barcode.FirebaseVisionBarcode
import com.google.firebase.ml.vision.cloud.FirebaseVisionCloudDetectorOptions
import com.google.firebase.ml.vision.cloud.landmark.FirebaseVisionCloudLandmark
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.document.FirebaseVisionCloudDocumentRecognizerOptions
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentText
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
import com.google.firebase.ml.vision.text.FirebaseVisionText
import kotlinx.android.synthetic.main.activity_main.*
import java.io.InputStream
import java.math.RoundingMode
import java.text.DecimalFormat
import java.util.*
import kotlin.system.measureNanoTime


class MainActivity : AppCompatActivity(), AdapterView.OnItemSelectedListener {

    private var imageId: Int = 1
    private var mlAPIId: Int = 0
    private val maxImageNum : Int = 50

    private var detTextMap : HashMap<Int, String>? = null

    //mlAPIId = 0
    private val detector = FirebaseVision.getInstance().onDeviceTextRecognizer
    private val imageDetectedTextMap : HashMap<Int, String> = HashMap(maxImageNum)
    private var det = detector

    //mlAPIId = 1
    private val detectorCloud = FirebaseVision.getInstance().cloudTextRecognizer
    private val imageDetectedTextMapCloud : HashMap<Int, String> = HashMap(maxImageNum)

    //private val languageIdentifier = FirebaseNaturalLanguage.getInstance().languageIdentification
    private val languageOptions = FirebaseLanguageIdentificationOptions.Builder()
        .setConfidenceThreshold(0.02f)
        .build()
    private val languageIdentifier = FirebaseNaturalLanguage.getInstance().getLanguageIdentification(languageOptions)
    private val imageLanguageMap : HashMap<Int, String> = HashMap(maxImageNum)

    //mlAPIId = 4
    private val optionsDocumentCloud = FirebaseVisionCloudDocumentRecognizerOptions.Builder()
        .setLanguageHints(Arrays.asList("en", "ja", "hi"))
        .build()
    private val detectorDocumentCloud = FirebaseVision.getInstance()
        .getCloudDocumentTextRecognizer(optionsDocumentCloud)

    private val imageDetectedDocumentMapCloud : HashMap<Int, String> = HashMap(maxImageNum)

    //mlAPIId = 2
    private val detectorBarCode  = FirebaseVision.getInstance().visionBarcodeDetector
    private val imageBarCodeTxtMap : HashMap<Int, String> = HashMap(maxImageNum)

    //mlAPIId = 3
    private val optionsLandmark = FirebaseVisionCloudDetectorOptions.Builder()
        .setModelType(FirebaseVisionCloudDetectorOptions.LATEST_MODEL)
        .setMaxResults(4)
        .build()
    private val detectorLandMark = FirebaseVision.getInstance().getVisionCloudLandmarkDetector(optionsLandmark)
    private val imageLandmarkTxtMap : HashMap<Int, String> = HashMap(maxImageNum)

    //mlAPIId = 5
    val highAccuracyOpts = FirebaseVisionFaceDetectorOptions.Builder()
        .setPerformanceMode(FirebaseVisionFaceDetectorOptions.ACCURATE)
        .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
        .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
        .build()
    val detectorFace = FirebaseVision.getInstance()
        .getVisionFaceDetector(highAccuracyOpts)
    private val imageFaceDetectionMap : HashMap<Int, String> = HashMap(maxImageNum)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val aa = ArrayAdapter(this, android.R.layout.simple_spinner_item, resources.getStringArray(R.array.mlmodel))
        aa.setDropDownViewResource(R.layout.support_simple_spinner_dropdown_item)
        mlAPIList.adapter = aa
        mlAPIList.onItemSelectedListener = this
    }

    override fun onItemSelected(arg0: AdapterView<*>, arg1: View, position: Int, id: Long) {
        Log.d("Spinner", "Inside Item Selected $position")
        mlAPIId = position
        when(mlAPIId) {
            0 -> det = detector
            1 -> det = detectorCloud
        }
        detTextMap = when(mlAPIId) {
            0 -> imageDetectedTextMap
            1 -> imageDetectedTextMapCloud
            2 -> imageBarCodeTxtMap
            3 -> imageLandmarkTxtMap
            4 -> imageDetectedDocumentMapCloud
            5 -> imageFaceDetectionMap
            else -> imageDetectedTextMap
        }
        setText()
        when(mlAPIId) {
            0,1,4 -> det_lang.visibility = View.VISIBLE
            else -> det_lang.visibility = View.INVISIBLE
        }
    }

    override fun onNothingSelected(arg0: AdapterView<*>) {
        Log.d("Spinner", "Inside No Item Selected")
        Toast.makeText(this, "Selected : 0", Toast.LENGTH_SHORT).show()
        mlAPIId = 0
        det = detector
        detTextMap = imageDetectedTextMap
        setText()
    }


    fun chooseNextImage(view: View) {
        imageId++
        imageId %= maxImageNum+1
        if (imageId == 0) {
            imageId = 1
        }
        Log.d("IMAGE CHOSEN", "$imageId.jpg")
        imageView2.setImageResource(getResource())
        setText()
    }

    fun choosePrevImage(view: View) {
        imageId--
        if (imageId == 0) {
            imageId = maxImageNum
        }
        imageId %= maxImageNum+1
        Log.d("IMAGE CHOSEN", "$imageId.jpg")
        imageView2.setImageResource(getResource())
        setText()
    }

    fun runDetector(view: View) {
        Log.d("FILENAME", "drawable/image_$imageId.jpg")
        val open: InputStream = this.applicationContext.resources.openRawResource(getResource())
        val bitmap: Bitmap = BitmapFactory.decodeStream(open)
        val image = FirebaseVisionImage.fromBitmap(bitmap)
        next_image.isEnabled = false
        prev_image.isEnabled = false
        detect_text.isEnabled = false
        val timeElapsed = measureNanoTime {
            when (mlAPIId) {
                0, 1 -> det.processImage(image)
                    .addOnSuccessListener { firebaseVisionText ->
                        Log.d("Text Recognition Call", "Success")
                        processTextRecognitionResult(firebaseVisionText)
                    }
                    .addOnFailureListener {
                        makeText(applicationContext, "No Text Detected", Toast.LENGTH_LONG).show()
                        Log.d("Text Recognition Call", "Failure")
                    }
                    .addOnCompleteListener {
                        next_image.isEnabled = true
                        prev_image.isEnabled = true
                        detect_text.isEnabled = true
                    }
                2 -> detectorBarCode.detectInImage(image)
                    .addOnSuccessListener { barcodes ->
                        Log.d("BarCode det Call", "Success")
                        processBarCodeResults(barcodes)
                    }
                    .addOnFailureListener {
                        makeText(applicationContext, "No Barcode Detected", Toast.LENGTH_LONG).show()
                        Log.d("Barcode Detection Call", "Failure")
                    }
                    .addOnCompleteListener {
                        next_image.isEnabled = true
                        prev_image.isEnabled = true
                        detect_text.isEnabled = true
                    }

                3 -> detectorLandMark.detectInImage(image)
                    .addOnSuccessListener { firebaseVisionCloudLandmarks ->
                        Log.d("Landmark det Call", "Success")
                        processLandMarkResults(firebaseVisionCloudLandmarks)
                    }
                    .addOnFailureListener {
                        makeText(applicationContext, "No Landmark Detected", Toast.LENGTH_LONG).show()
                        Log.d("Landmark Det Call", "Failure")
                    }
                    .addOnCompleteListener {
                        next_image.isEnabled = true
                        prev_image.isEnabled = true
                        detect_text.isEnabled = true
                    }
                4 -> detectorDocumentCloud.processImage(image)
                    .addOnSuccessListener { firebaseVisionDocumentText ->
                        Log.d("Doc Text Recog Call", "Success")
                        processDocumentRecognitionResult(firebaseVisionDocumentText)
                    }
                    .addOnFailureListener {
                        makeText(applicationContext, "No Text Detected in Document", Toast.LENGTH_LONG).show()
                        Log.d("Doc Text Recog Call", "Failure")
                    }
                    .addOnCompleteListener {
                        next_image.isEnabled = true
                        prev_image.isEnabled = true
                        detect_text.isEnabled = true
                    }
                5 -> detectorFace.detectInImage(image)
                    .addOnSuccessListener { faces ->
                        Log.d("Doc Text Recog Call", "Success")
                        processFaceDetectionResult(faces)
                    }
                    .addOnFailureListener {
                        makeText(applicationContext, "No Face Detected in Image", Toast.LENGTH_LONG).show()
                        Log.d("Face Detection Call", "Failure")
                    }
                    .addOnCompleteListener {
                        next_image.isEnabled = true
                        prev_image.isEnabled = true
                        detect_text.isEnabled = true
                    }

            }
        }
        Log.d("PERF", "Detector took $timeElapsed nanosec to run")

    }

    private fun getResource(): Int {
        return applicationContext.resIdByName("image_$imageId", "drawable")
    }

    private fun setText() {
        if (detTextMap?.containsKey(imageId) == true) {
            detected_text.text = detTextMap!![imageId]
        }
        else detected_text.text = ""

        if (imageLanguageMap.containsKey(imageId)) {
            det_lang.text = imageLanguageMap[imageId]
        }
        else det_lang.text = "Language"
    }

    private fun processDocumentRecognitionResult(texts: FirebaseVisionDocumentText) {
        val blocks = texts.blocks
        if (blocks.size == 0) {
            makeText(applicationContext, "No Text Detected in Document", Toast.LENGTH_LONG).show()
            imageDetectedDocumentMapCloud[imageId] = "No Text Detected on Image"
            detected_text.text = imageDetectedDocumentMapCloud[imageId]
        }

        var result = ""
        for (block in blocks) {
            for (paragraph in block.paragraphs) {
                for (word in paragraph.words) {
                    for (symbol in word.symbols) {
                        result += symbol.text
                    }
                    result += " "
                }
                result += "\n"
            }
        }
        if (result != "") {
            makeText(applicationContext, "Text Detected", Toast.LENGTH_SHORT).show()
            getLanguage(result)
            imageDetectedDocumentMapCloud[imageId] = result
            detected_text.text = imageDetectedDocumentMapCloud[imageId]
            Log.d("RETURN VALUE", "true $result")
        }
        else
        {
            imageDetectedDocumentMapCloud[imageId] = "No Text Detected in Image"
            detected_text.text = imageDetectedDocumentMapCloud[imageId]
            Log.d("RETURN VALUE", "false")
        }
    }

        private fun processTextRecognitionResult(texts: FirebaseVisionText) {
        val blocks = texts.textBlocks
        if (blocks.size == 0) {
            makeText(applicationContext, "No Text Detected", Toast.LENGTH_LONG).show()
            detTextMap!![imageId] = "No Text Detected on Image"
            detected_text.text = detTextMap!![imageId]
        }
        var result = ""
        for (i in blocks.indices) {
            val lines = blocks[i].lines
            for (j in lines.indices) {
                val elements = lines[j].elements
                for (k in elements.indices) {
                    Log.d("Text Recognition", elements[k]?.text ?: "")
                    result += elements[k]?.text + " "
                }
                result += "\t"
            }
            result += "\n"
        }
        if (result != "") {
            makeText(applicationContext, "Text Detected", Toast.LENGTH_SHORT).show()
            getLanguage(result)
            detTextMap!![imageId] = result
            detected_text.text = detTextMap!![imageId]
        }
        else
        {
            detTextMap!![imageId] = "No Text Detected in Image"
            detected_text.text = detTextMap!![imageId]
            Log.d("RETURN VALUE", "false")
        }
    }

    private fun getLanguage(result : String) {
        var res = "unrecognized"
        val df = DecimalFormat("#.##")
        df.roundingMode = RoundingMode.CEILING

        languageIdentifier.identifyPossibleLanguages(result)
             .addOnSuccessListener { languages ->
                 Log.d("Language Ident. Call", "Success")
                 for (language in languages) {
                     res = language.languageCode + ", conf = " + df.format(language.confidence).toString()
                     break
                 }
                 imageLanguageMap[imageId] = res
             }
            .addOnFailureListener {
                makeText(applicationContext, "No Language Identified", Toast.LENGTH_LONG).show()
                imageLanguageMap[imageId] = res
                Log.d("Language Ident. Call", "Failure")
            }
            .addOnCompleteListener { det_lang.text = res }
        }

    private fun processBarCodeResults (barcodes : MutableList<FirebaseVisionBarcode>) {
        var result = ""
        for (barcode in barcodes) {
            result += barcode.rawValue
            result += "\n"
        }
        val length = barcodes.size
        if (result != "") {
            makeText(applicationContext, "$length BarCode Detected", Toast.LENGTH_LONG).show()
            imageBarCodeTxtMap[imageId] = result
            detected_text.text = result
        }
        else
        {
            imageBarCodeTxtMap[imageId] = "No Barcode detected on Image"
            detected_text.text = imageBarCodeTxtMap[imageId]
        }
    }

    private fun processLandMarkResults (landmarks : MutableList<FirebaseVisionCloudLandmark>) {
        var result = ""
        for (landmark in landmarks) {
            result += landmark.landmark
            result += "\n"
        }
        val length = landmarks.size
        if (result != "") {
            makeText(applicationContext, "$length Landmark(s) Detected", Toast.LENGTH_LONG).show()
            imageLandmarkTxtMap[imageId] = result
            detected_text.text = result
            Log.d("RETURN VALUE", "true $result")
        }
        else
        {
            imageLandmarkTxtMap[imageId] = "No Landmark detected on Image"
            detected_text.text = imageLandmarkTxtMap[imageId]
        }
    }
    private fun processFaceDetectionResult(faces : MutableList<FirebaseVisionFace>) {
        val length = faces.size
        var result = "Number of faces detected = " + faces.size.toString() + "\n"
        if (faces.size > 0) {
            result += "Smiling Probabilities: "
        }
        for (face in faces) {
            result += face.smilingProbability.toString() + ", "
        }
        if (result != "") {
            makeText(applicationContext, "$length Faces Detected", Toast.LENGTH_LONG).show()
            imageFaceDetectionMap[imageId] = result
            detected_text.text = result
            Log.d("RETURN VALUE", "true $result")
        }
        else
        {
            imageFaceDetectionMap[imageId] = "No Face detected on Image"
            detected_text.text = imageLandmarkTxtMap[imageId]
        }
    }
}

fun Context.resIdByName(resIdName: String?, resType: String): Int {
    resIdName?.let {
        return resources.getIdentifier(it, resType, packageName)
    }
    throw Resources.NotFoundException()
}