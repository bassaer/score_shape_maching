//
//  main.cpp
//  score_shape_maching
//
//  Created by Nakayama on 2016/06/16.
//  Copyright © 2016年 Nakayama. All rights reserved.
//

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

const double huMomentThres = 0.005;  // 形状マッチングの閾値
const float IMG_SIZE = 0.4;

cv::Mat changeOrientation(cv::Mat mat){
    cv::Mat result = cv::Mat(mat.rows,mat.cols, CV_8UC4);
    cv::transpose(mat, result);
    //cv::flip(mat, result, 0);
    return result;
}

void rotate_90n(cv::Mat &src, cv::Mat &dst, int angle)
{
    dst.create(src.size(), src.type());
    if(angle == 270 || angle == -90){
        // Rotate clockwise 270 degrees
        cv::transpose(src, dst);
        cv::flip(dst, dst, 0);
    }else if(angle == 180 || angle == -180){
        // Rotate clockwise 180 degrees
        cv::flip(src, dst, -1);
    }else if(angle == 90 || angle == -270){
        // Rotate clockwise 90 degrees
        cv::transpose(src, dst);
        cv::flip(dst, dst, 1);
    }else if(angle == 360 || angle == 0){
        if(src.data != dst.data){
            src.copyTo(dst);
        }
    }
}

void convertBinImage(cv::Mat src, cv::Mat &dst){
    cv::threshold(src, dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //cv::adaptiveThreshold(dst, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 7, 8);
}



void doShapesMatching(cv::Mat &src, cv::Mat &temp){
    //画像を回転
    //rotate_90n(src, src, 270);
    //src = changeOrientation(src);
    
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, CV_BGR2GRAY);
    
    // 二値化
    cv::Mat temp_bin, src_bin;
    
    convertBinImage(src_gray, src_bin);
    convertBinImage(temp, temp_bin);
    
    // ラベリング
    cv::Mat labelsImg;
    cv::Mat stats;
    cv::Mat centroids;
    int nLabels = cv::connectedComponentsWithStats(src_bin, labelsImg, stats, centroids);
    
    // ラベルごとのROIを得る(0番目は背景なので無視)
    cv::Mat roiImg;
    cv::cvtColor(src_bin, roiImg, CV_GRAY2BGR);
    std::vector<cv::Rect> roiRects;
    for (int i = 1; i < nLabels; i++) {
        int *param = stats.ptr<int>(i);
        
        int x = param[cv::ConnectedComponentsTypes::CC_STAT_LEFT];
        int y = param[cv::ConnectedComponentsTypes::CC_STAT_TOP];
        int height = param[cv::ConnectedComponentsTypes::CC_STAT_HEIGHT];
        int width = param[cv::ConnectedComponentsTypes::CC_STAT_WIDTH];
        roiRects.push_back(cv::Rect(x, y, width, height));
        
        cv::rectangle(roiImg, roiRects.at(i-1), cv::Scalar(0, 255, 0), 2);
    }
    
    // huモーメントによる形状マッチングを行う
    cv::Mat dst = src.clone();
    for (int i = 1; i < nLabels; i++){
        cv::Mat roi = src_bin(roiRects.at(i-1));    // 対象とするブロブの領域取り出し
        double similarity = cv::matchShapes(temp_bin, roi, CV_CONTOURS_MATCH_I3, 0);    // huモーメントによるマッチング
        
        if (similarity < huMomentThres){
            printf("similarity = %f\n",similarity);
            if(similarity < 0.0005){
                cv::rectangle(dst, roiRects.at(i - 1), cv::Scalar(0, 255, 0), 4);
            }else{
                cv::rectangle(dst, roiRects.at(i - 1), cv::Scalar(255, 0, 0), 4);
            }
            
        }
    }
    
    cv::resize(src, src, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::resize(dst, dst, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::resize(src_bin, src_bin, cv::Size(), IMG_SIZE, IMG_SIZE);
    //cv::resize(temp_bin, temp_bin, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::imshow("template", temp);
    //cv::imshow("src", src);
    cv::imshow("temp_bin", temp_bin);
    cv::imshow("src_bin", src_bin);
    cv::imshow("dst", dst);
    
    cv::waitKey();
    
    cv::imwrite("dst.jpg", dst);
}

void doTemplateMatching(cv::Mat &src, cv::Mat &temp){
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, CV_BGR2GRAY);
    
    // 二値化
    cv::Mat temp_bin, src_bin;
    
    convertBinImage(src_gray, src_bin);
    convertBinImage(temp, temp_bin);
    
    cv::Mat result;
    cv::matchTemplate(src_bin, temp_bin, result, CV_TM_CCORR_NORMED);
    
    std::vector<cv::Point> detected_points;
    float threshold = 0.96f;
    for(int y=0;y<result.rows;y++){
        for(int x=0;x<result.cols;x++){
            if(result.at<float>(y,x) > threshold){
                detected_points.push_back(cv::Point(x, y));
            }
        }
    }
    
    //表示
    cv::Mat dst = src.clone();
    for (auto it = detected_points.begin(); it != detected_points.end(); ++it){
        cv::rectangle(dst, *it, cv::Point(it->x+temp.cols,it->y+temp.rows), cv::Scalar(0,255,0), 2, 8, 0);
    }
    
    cv::resize(src, src, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::resize(dst, dst, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::resize(src_bin, src_bin, cv::Size(), IMG_SIZE, IMG_SIZE);
    //cv::resize(temp_bin, temp_bin, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::imshow("template", temp);
    //cv::imshow("src", src);
    cv::imshow("temp_bin", temp_bin);
    cv::imshow("src_bin", src_bin);
    cv::imshow("dst", dst);
    
    cv::waitKey();
    
}

void doCascadeMatching(cv::Mat &src) {
    cv::Mat src_gray;
    rotate_90n(src, src, 270);
    cv::cvtColor(src, src_gray, CV_BGR2GRAY);
    
    // 二値化
    cv::Mat src_bin;
    convertBinImage(src_gray, src_bin);
    
    std::vector<cv::Rect> results;
    //cv::CascadeClassifier cascade("/Users/nakayama/Desktop)
    cv::CascadeClassifier cascade("/Users/nakayama/Desktop/cascade/2/cascade.xml");
    //cv::CascadeClassifier test("/usr/local/Cellar/opencv3/3.1.0_3/share/OpenCV/haarcascades/haarcascade_frontalcatface.xml");
    //Cascadeで検出
    cv::Mat dst = src.clone();
    
    cascade.detectMultiScale(src_bin, results, 1.1, 2);
    for(auto it = results.begin(); it != results.end(); ++it){
        cv::rectangle(dst, it->tl(), it->br(), cv::Scalar(0, 255, 0), 2,8, 0);
    }
    
    cv::resize(src, src, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::resize(dst, dst, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::resize(src_bin, src_bin, cv::Size(), IMG_SIZE, IMG_SIZE);
    cv::imshow("src_bin", src_bin);
    cv::imshow("dst", dst);

    cv::waitKey();
    
}

int main()
{
    cv::Mat temp = cv::imread("/Users/nakayama/Desktop/pattern.png", 0); //
    if (temp.empty()){
        std::cout << "テンプレート読み込みエラー" << std::endl;
        return -1;
    }
    
    //cv::Mat src = cv::imread("/Users/nakayama/Desktop/mark_sheet.png"); // 撮影された画像
    cv::Mat src = cv::imread("/Users/nakayama/Desktop/camera.JPG");
    //cv::Mat src = cv::imread("/Users/nakayama/Desktop/dammy.png");
    if (src.empty()){
        std::cout << "入力画像読み込みエラー" << std::endl;
        return -1;
    }
    
    //doShapesMatching(src, temp);
    
    //doTemplateMatching(src, temp);
    
    doCascadeMatching(src);
    
    
    return 0;
}

