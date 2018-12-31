package opencv.hegi.countpeopleopencv;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;


public class OpenCvController extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    int method = 0;

    private Mat mRgba;
    private Mat mRgba2;
    private Mat mGray;
    private Mat mGray2;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;


    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    int counterUp = 0;
    int counterDown = 0;
    int counterFrames = 0;
    int counterRefresh = 0;
    double xCenter = -1;
    double yCenter = -1;

    // ======== TRACKING DEFINITION VARIABLES ===================== //
    private ArrayList<PersonCoordinate> personCoordinates;   // This array provide coordenades of a real frame
    private ArrayList<PersonCoordinate> personTestCoordinates;  // This array help us to see if there are noise in frame

    private int zone1 = 0;
    private int zone2 = 0;
    private int zone3 = 0;
    private int zone4 = 0;
    private int zone5 = 0;
    private int zone6 = 0;
    private int zone7 = 0;
    private int zone8 = 0;

    private int widthRec = 0;
    private int widthRecSaved = 0;
    private int widthResolution = 0;
    private int heigthResolution = 0;

    private int limitZones = 0;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        // Carga de Haarcascade Profileface haarcascade_profileface.xml
                        InputStream is = getResources().openRawResource(R.raw.haarcascade_profileface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "haarcascade_profileface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[1024];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.setCameraIndex(0);
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public OpenCvController() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        personCoordinates = new ArrayList<PersonCoordinate>();
        personTestCoordinates = new ArrayList<PersonCoordinate>();
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_open_cv);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        widthResolution = width;
        heigthResolution = height;
        limitZones = widthResolution/8;
        mGray = new Mat();
        mRgba = new Mat();
        mRgba2 = new Mat();
        mGray2 = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Core.flip(mGray, mGray2, 1);
        Core.flip(mRgba, mRgba2, 1);

        int x1 = (limitZones*11)/2 ;
        int y1 = heigthResolution;
        int x2 = (limitZones*5)/2 ;
        int y2 = heigthResolution;

        // se dibujan las Zonas de desición
        Imgproc.line(mRgba, new Point(limitZones*1, 0), new Point(limitZones*1, y1), new Scalar(255, 255, 0), 1);
        Imgproc.line(mRgba, new Point(limitZones*2, 0), new Point(limitZones*2, y1), new Scalar(255, 255, 0), 1);
        Imgproc.line(mRgba, new Point(limitZones*3, 0), new Point(limitZones*3, y1), new Scalar(255, 255, 0), 1);
        Imgproc.line(mRgba, new Point(limitZones*4, 0), new Point(limitZones*4, y1), new Scalar(255, 255, 0), 1);
        Imgproc.line(mRgba, new Point(limitZones*5, 0), new Point(limitZones*5, y1), new Scalar(255, 255, 0), 1);
        Imgproc.line(mRgba, new Point(limitZones*6, 0), new Point(limitZones*6, y1), new Scalar(255, 255, 0), 1);
        Imgproc.line(mRgba, new Point(limitZones*7, 0), new Point(limitZones*7, y1), new Scalar(255, 255, 0), 1);


        Imgproc.line(mRgba, new Point(x1, 0), new Point(x1, y1), new Scalar(255, 0, 0), 3);
        Imgproc.line(mRgba, new Point(x2, 0), new Point(x2, y2), new Scalar(0, 255, 0), 3);

        Imgproc.putText(mRgba, "Up: " + counterUp,
                new Point(20, 60),
                Core.FONT_HERSHEY_SIMPLEX, 1.6, new Scalar(0, 255, 0,
                        255));

        Imgproc.putText(mRgba, "Down: " + counterDown,
                new Point(1050, 60),
                Core.FONT_HERSHEY_SIMPLEX, 1.6, new Scalar(255, 0, 0,
                        255));

        Imgproc.putText(mRgba, "Frames: " + counterFrames,
                new Point(20, 120),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        // =====================CONTADORES DE ZONAS ===========================================//
        Imgproc.putText(mRgba, "Zona1: " + zone1,
                new Point(20, 150),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "Zona2: " + zone2,
                new Point(20, 180),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "Zona3: " + zone3,
                new Point(20, 210),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "Zona4: " + zone4,
                new Point(20, 240),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "Zona5: " + zone5,
                new Point(20, 270),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "Zona6: " + zone6,
                new Point(20, 300),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "Zona7: " + zone7,
                new Point(20, 330),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "Zona8: " + zone8,
                new Point(20, 360),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "WIDTH: " + widthRec,
                new Point(20, 390),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        Imgproc.putText(mRgba, "REFRESH: " + counterRefresh,
                new Point(20, 420),
                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 0,
                        255));

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();
        MatOfRect faces2 = new MatOfRect();


        // Left
        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                Log.e("detector1", "Entra a detector1");
            mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else {
            Log.e(TAG, "Detection method is not selected!");
        }

        // Perfil de rostro

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                Log.e("detector2", "Entra a detector2");
            mJavaDetector.detectMultiScale(mGray2, faces2, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else {
            Log.e(TAG, "Detection method is not selected!");
        }


        Rect[] facesArray = faces.toArray();
        Rect[] facesArray2 = faces2.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            Log.e("FacesArray", String.valueOf(facesArray.length));
            widthRec = facesArray[i].width;
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(),
                    FACE_RECT_COLOR, 3);
            xCenter = (facesArray[i].x + facesArray[i].width + facesArray[i].x) / 2;
            yCenter = (facesArray[i].y + facesArray[i].y + facesArray[i].height) / 2;
            Point center = new Point(xCenter, yCenter);

            Imgproc.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3);

            Imgproc.putText(mRgba, "[" + center.x + "," + center.y + "]",
                    new Point(center.x + 20, center.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));

            // La variable myPersonCoordinate guarda las coordenadas X,Y por cada ciclo de detección
            PersonCoordinate myPersonCoordinate = new PersonCoordinate();


            myPersonCoordinate.setHorizontal(facesArray[i].x);
            myPersonCoordinate.setVertical(facesArray[i].y);


            if (personCoordinates.size() != 0) {
                // personCoordinates.add(myPersonCoordinate);
            } else {
                if (personTestCoordinates.size() == 0) {
                    personTestCoordinates.add(myPersonCoordinate);
                } else {

                    // In this else it is necessary to evaluate if the previous detected frame has in common
                    // similar coordinates with the new capture
                    int sizeArray = 1;

                    int lastPosition = 0;
                    lastPosition = personTestCoordinates.size() - 1;
                    int lastVertical = personTestCoordinates.get(lastPosition).getVertical();  // last value of the horizontal coordinate
                    int lastHorizontal = personTestCoordinates.get(lastPosition).getHorizontal(); // last value of the vertical coordinate
                    int actualVertical = myPersonCoordinate.getVertical();
                    int actualHorizontal = myPersonCoordinate.getHorizontal();

                    // This conditional determine if the actual vertical value is near of the pervious value saved
                    if (actualVertical >= lastVertical - 80 && actualVertical <= lastVertical + 80 && actualHorizontal >= lastHorizontal - 80 && actualHorizontal <= lastHorizontal + 80) {
                        if (actualHorizontal > limitZones*7) {
                            zone8 = 1;
                        }
                        if (actualHorizontal > limitZones*6 && actualHorizontal < limitZones*7) {
                            zone7 = 1;
                        }
                        if (actualHorizontal > limitZones*5 && actualHorizontal < limitZones*6) {
                            zone6 = 1;
                        }
                        if (actualHorizontal > limitZones*4 && actualHorizontal < limitZones*5) {
                            zone5 = 1;
                        }
                        if (actualHorizontal > limitZones*3 && actualHorizontal < limitZones*4) {
                            zone4 = 1;
                        }
                        if (actualHorizontal > limitZones*2 && actualHorizontal < limitZones*3) {
                            zone3 = 1;
                        }
                        if (actualHorizontal > limitZones*1 && actualHorizontal < limitZones*2) {
                            zone2 = 1;
                        }
                        if (actualHorizontal < limitZones*1) {
                            zone1 = 1;
                        }

                        // Here comes coordinated which belongs to the real object detected
                        counterFrames++;
                        personTestCoordinates.add(myPersonCoordinate);
                        if (actualHorizontal < (limitZones*5)/2) {
                            evaluateUpPassager();
                            // function to evaluate
                        }

                        if (actualHorizontal > (limitZones*11)/2) {
                            evaluateDownPassager();
                            // function to evaluate
                        }

                        widthRecSaved = widthRec;

                    } else {

                        counterRefresh++;

                        if (counterRefresh > 30) {
                            personTestCoordinates.clear();
                            counterRefresh = 0;
                            counterFrames = 0;
                            zone1 = 0;
                            zone2 = 0;
                            zone3 = 0;
                            zone4 = 0;
                            zone5 = 0;
                            zone6 = 0;
                            zone7 = 0;
                            zone8 = 0;
                        }
                        // Un contador que si llega a cierto numero dispara el evento de limpiar el array

                    }
                }
            }

        }

        for (int i = 0; i < facesArray2.length; i++) {
            int xx = widthResolution - facesArray2[i].x;
            double xbien = Math.abs(xx);
            Log.e("FacesArrayX", String.valueOf(xbien));
            Point tlAbs = new Point(Math.abs(widthResolution - facesArray2[i].tl().x), facesArray2[i].tl().y);
            Point brAbs = new Point(Math.abs(widthResolution - facesArray2[i].br().x), facesArray2[i].br().y);
            Imgproc.rectangle(mRgba, tlAbs, brAbs,
                    FACE_RECT_COLOR, 3);
            xCenter = xbien - (facesArray2[i].width) / 2;
            yCenter = (facesArray2[i].y + facesArray2[i].y + facesArray2[i].height) / 2;
            Point center = new Point(xCenter, yCenter);

            Imgproc.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 3);

            Imgproc.putText(mRgba, "[" + center.x + "," + center.y + "]",
                    new Point(center.x + 20, center.y + 20),
                    Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255,
                            255));

            // La variable myPersonCoordinate guarda las coordenadas X,Y por cada ciclo de detección
            PersonCoordinate myPersonCoordinate = new PersonCoordinate();


            myPersonCoordinate.setHorizontal(xx);
            myPersonCoordinate.setVertical(facesArray2[i].y);


            if (personCoordinates.size() != 0) {
                // personCoordinates.add(myPersonCoordinate);
            } else {
                if (personTestCoordinates.size() == 0) {
                    personTestCoordinates.add(myPersonCoordinate);
                } else {

                    // In this else it is necessary to evaluate if the previous detected frame has in common
                    // similar coordinates with the new capture
                    int sizeArray = 1;

                    int lastPosition = 0;
                    lastPosition = personTestCoordinates.size() - 1;
                    int lastVertical = personTestCoordinates.get(lastPosition).getVertical();  // last value of the horizontal coordinate
                    int lastHorizontal = personTestCoordinates.get(lastPosition).getHorizontal(); // last value of the vertical coordinate
                    int actualVertical = myPersonCoordinate.getVertical();
                    int actualHorizontal = myPersonCoordinate.getHorizontal();

                    // This conditional determine if the actual vertical value is near of the pervious value saved
                    if (actualVertical >= lastVertical - 80 && actualVertical <= lastVertical + 80 && actualHorizontal >= lastHorizontal - 80 && actualHorizontal <= lastHorizontal + 80) {
                        if (actualHorizontal > limitZones*7) {
                            zone8 = 1;
                        }
                        if (actualHorizontal > limitZones*6 && actualHorizontal < limitZones*7) {
                            zone7 = 1;
                        }
                        if (actualHorizontal > limitZones*5 && actualHorizontal < limitZones*6) {
                            zone6 = 1;
                        }
                        if (actualHorizontal > limitZones*4 && actualHorizontal < limitZones*5) {
                            zone5 = 1;
                        }
                        if (actualHorizontal > limitZones*3 && actualHorizontal < limitZones*4) {
                            zone4 = 1;
                        }
                        if (actualHorizontal > limitZones*2 && actualHorizontal < limitZones*3) {
                            zone3 = 1;
                        }
                        if (actualHorizontal > limitZones*1 && actualHorizontal < limitZones*2) {
                            zone2 = 1;
                        }
                        if (actualHorizontal < limitZones*1) {
                            zone1 = 1;
                        }

                        // Here comes coordinated which belongs to the real object detected
                        counterFrames++;
                        personTestCoordinates.add(myPersonCoordinate);
                        if (actualHorizontal < (limitZones*5)/2) {
                            evaluateUpPassager();
                            // function to evaluate
                        }

                        if (actualHorizontal > (limitZones*11)/2) {
                            evaluateDownPassager();
                            // function to evaluate
                        }

                        widthRecSaved = widthRec;

                    } else {

                        counterRefresh++;

                        if (counterRefresh > 30) {
                            personTestCoordinates.clear();
                            counterRefresh = 0;
                            counterFrames = 0;
                            zone1 = 0;
                            zone2 = 0;
                            zone3 = 0;
                            zone4 = 0;
                            zone5 = 0;
                            zone6 = 0;
                            zone7 = 0;
                            zone8 = 0;

                        }
                        // Un contador que si llega a cierto numero dispara el evento de limpiar el array

                    }
                }
            }

        }


        return mRgba;
    }

    // In this function it is validated if the person made the trip to get off the bus
    public void evaluateDownPassager() {

        int average = (zone1 + zone2 + zone3 + zone4 + zone5);
        if (average >= 1) {
            counterDown++;
            personTestCoordinates.clear();
            counterFrames = 0;
            counterRefresh = 0;
            zone1 = 0;
            zone2 = 0;
            zone3 = 0;
            zone4 = 0;
            zone5 = 0;
            zone6 = 0;
            zone7 = 0;
            zone8 = 0;
        }
    }

    public void evaluateUpPassager() {

        int average = (zone8 + zone7 + zone6 + zone5 + zone4);
        if (average >= 1) {
            counterUp++;
            personTestCoordinates.clear();
            counterFrames = 0;
            counterRefresh = 0;
            zone1 = 0;
            zone2 = 0;
            zone3 = 0;
            zone4 = 0;
            zone5 = 0;
            zone6 = 0;
            zone7 = 0;
            zone8 = 0;
        }
    }


}
