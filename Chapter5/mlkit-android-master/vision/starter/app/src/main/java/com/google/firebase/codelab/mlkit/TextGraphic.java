// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.firebase.codelab.mlkit;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.codelab.mlkit.GraphicOverlay.Graphic;

/**
 * Graphic instance for rendering TextBlock position, size, and ID within an associated graphic
 * overlay view.
 */
public class TextGraphic extends Graphic {

    private static final String TAG = "TextGraphic";
    private static final int TEXT_COLOR = Color.RED;
    private static final int CHAR_COLOR = Color.BLUE;  //ashish
    private static final float TEXT_SIZE = 54.0f;
    private static final float STROKE_WIDTH = 4.0f;

    private final Paint rectPaint;
    private final Paint textPaint;
    private final Paint charPaint;  //ashish

    private final FirebaseVisionText.Element element;

    TextGraphic(GraphicOverlay overlay, FirebaseVisionText.Element element) {
        super(overlay);

        this.element = element;

        rectPaint = new Paint();
        rectPaint.setColor(TEXT_COLOR);
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(STROKE_WIDTH);

        textPaint = new Paint();
        textPaint.setColor(TEXT_COLOR);
        textPaint.setTextSize(TEXT_SIZE);

        // added by ashish
        charPaint = new Paint();
        charPaint.setColor(CHAR_COLOR);
        charPaint.setStyle(Paint.Style.STROKE);
        charPaint.setStrokeWidth(STROKE_WIDTH);


        // Redraw the overlay, as this graphic has been added.
        postInvalidate();
    }

    /**
     * Draws the text block annotations for position, size, and raw value on the supplied canvas.
     */
    @Override
    public void draw(Canvas canvas) {
        Log.d(TAG, "on draw text graphic");
        if (element == null) {
            throw new IllegalStateException("Attempting to draw a null text.");
        }

        // Draws the bounding box around the TextBlock.
        RectF rect = new RectF(element.getBoundingBox());
        //canvas.drawRect(rect, rectPaint);

        // Renders the text at the bottom of the box.
        canvas.drawText(element.getText(), rect.left, rect.bottom, textPaint);

        // now render boxes around characters
        drawChars(canvas, rect);

    }

    /**
     * Added by Ashish: To draw bounding boxes around characters
     * @param canvas
     * @param word
     */
    public void drawChars(Canvas canvas, RectF word) {
        Log.d(TAG, "on drw chars text graphic");
        // this method should be called from inside the draw canvas
        // method after basic drawing is done

        // attempt to create bounding boxes for each character from the word level box
        // should only work for latin languages like English
        float width = word.right - word.left;

        int numChars = element.getText().length();

        float charWidth = width / numChars;
        // now create equi-spaced char rectangles

        float left = word.left;
        float right = left;
        for(int i=0; i < numChars; i++) {
            right = (left + charWidth) - 1;
            RectF charX = new RectF(left, word.top, right, word.bottom);
            canvas.drawRect(charX, charPaint);

            left = left + charWidth;
        }


    }
}
