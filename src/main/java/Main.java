import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ximgproc.StructuredEdgeDetection;
import org.opencv.ximgproc.Ximgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Main {
    static {System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    public static void getRect(String fileName) {

        String folderPath = "/usr/local/hicoden/test/";

        List<Double> x_coords = new ArrayList<>();
        List<Double> y_coords = new ArrayList<>();

        Mat rectangle = Imgcodecs.imread(folderPath + fileName + ".png");
        int imgArea = rectangle.width() * rectangle.height();
        Mat img = rectangle.clone();

//        Mat imgGray = new Mat();
//
//        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2GRAY);
//        Imgcodecs.imwrite(folderPath + fileName + "-BGR2GRAY" + ".png", imgGray);
//
//        Imgproc.GaussianBlur(imgGray, imgGray, new Size(5, 5), 0);
//        Imgcodecs.imwrite(folderPath + fileName + "-gaussianBlur" + ".png", imgGray);
//
//        Mat threshedImg = new Mat();
////        Imgproc.adaptiveThreshold(imgGray, threshedImg, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc
////                .THRESH_BINARY, 11, 2);
////        Imgcodecs.imwrite(folderPath + fileName + "-adaptiveThreshold" + ".png", threshedImg);
//
//        Imgproc.threshold(imgGray, threshedImg, 0, 255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
//        Imgcodecs.imwrite(folderPath + fileName + "-binaryThreshold" + ".png", threshedImg);
//
        List<MatOfPoint> contours = new ArrayList<>();
//        Mat hierarchy = new Mat();
//
//        Imgproc.findContours(threshedImg, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
//        Imgcodecs.imwrite(folderPath + fileName + "-contours" + ".png", hierarchy);

        StructuredEdgeDetection pDollar = Ximgproc.createStructuredEdgeDetection("/usr/local/hicoden/test/model.yml");
        Mat src = new Mat();
        img.convertTo(src, CvType.CV_32F, 1.0 / 255.0);
        Mat edges = new Mat(src.size(), src.type());
        pDollar.detectEdges(src, edges);
        Mat output = new Mat();
        edges.convertTo(output, CvType.CV_8UC1, 255.0);
        Imgproc.findContours(output, contours, new Mat(),Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        double max_area = 0;
        int num = 0;

        for (int i = 0; i < contours.size(); i++) {

            double area = Imgproc.contourArea(contours.get(i));

            // if area is smaller than screen capture area
            if (area > 100 && area < imgArea / 1.2) {
                MatOfPoint2f mop = new MatOfPoint2f(contours.get(i).toArray());
                double peri = Imgproc.arcLength(mop, true);

                MatOfPoint2f approx = new MatOfPoint2f();
                System.out.println(peri);
                Double epsilon = peri * 0.02;
                System.out.println(epsilon);
                Imgproc.approxPolyDP(mop, approx, epsilon, true);

                if (area > max_area && approx.toArray().length == 4) {
                    MatOfPoint2f biggest = approx;
                    double[] pointsDouble;

                    pointsDouble = biggest.get(0, 0);
                    Point p1 = new Point(pointsDouble[0], pointsDouble[1]);

                    pointsDouble = biggest.get(1, 0);
                    Point p2 = new Point(pointsDouble[0], pointsDouble[1]);

                    pointsDouble = biggest.get(2, 0);
                    Point p3 = new Point(pointsDouble[0], pointsDouble[1]);

                    pointsDouble = biggest.get(3, 0);
                    Point p4 = new Point(pointsDouble[0], pointsDouble[1]);


                    x_coords.add(p1.x);
                    x_coords.add(p2.x);
                    x_coords.add(p3.x);
                    x_coords.add(p4.x);

                    y_coords.add(p1.y);
                    y_coords.add(p2.y);
                    y_coords.add(p3.y);
                    y_coords.add(p4.y);

                    num = i;
                    max_area = area;
                }
            }
        }

        int x_min = new Double(Collections.min(x_coords)).intValue();
        int x_max = new Double(Collections.max(x_coords)).intValue();
        int y_min = new Double(Collections.min(y_coords)).intValue();
        int y_max = new Double(Collections.max(y_coords)).intValue();

        rectangle = Imgcodecs.imread(folderPath + fileName + ".png");
        img = rectangle.clone();

        Rect rect = new Rect(x_min, y_min, x_max - x_min, y_max - y_min);
        Mat croppedMat = new Mat(img, rect);

        Imgproc.drawContours(img, contours, num, new Scalar(0,0,255), 3);
        Imgcodecs.imwrite(folderPath + fileName + "-contoursDrawn" + ".png", img);
        Imgcodecs.imwrite(folderPath + fileName + "-cropped" + ".png", croppedMat);

    }

    public static void main(String[] args) {
        getRect(args[0]);
    }
}
