// Darknet.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "pch.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

#include "yolo_v2_class.hpp"


const char* cfg = "..\\..\\yolov3.cfg";
const char* weights = "..\\..\\yolov3.weights";
const char* names = "..\\..\\coco.names";

std::vector<std::string> classes;

std::vector<std::string> objects_names(const char* filename)
{
	std::vector<std::string> classes;

	FILE* fp = fopen(filename, "r");
	if (fp != NULL)
	{
		char buffer[80];
		while (fgets(buffer, sizeof(buffer), fp) != NULL)
		{
			char* eol = strrchr(buffer, '\n');
			if (eol != NULL)
				*eol = 0;
			classes.push_back(buffer);
		}
		fclose(fp);
	}
	return classes;
}

void drawObject(cv::Mat& image, bbox_t& object)
{
	if (object.prob < 0.6f)
		return;
	cv::Rect rect(object.x, object.y, object.w, object.h);
	cv::Point pos(object.x, object.y);
	cv::rectangle(image, rect, cv::Scalar(255 * object.prob, 0, 0), 2);
	cv::putText(image, classes[object.obj_id], pos, 0, 0.5, cv::Scalar(255, 255, 255));
}

int main()
{
	classes = objects_names(names);
	Detector darknet(cfg, weights);
	cv::VideoCapture camera;
	if (camera.open(0))
	{
		cv::Mat image;
		while (camera.grab())
		{
			camera >> image;
			std::vector<bbox_t> objects = darknet.detect(image);
			for (size_t i = 0; i < objects.size(); i++)
			{
				drawObject(image, objects[i]);
			}
			cv::imshow("Darknet", image);
		}
		camera.release();
	}
}

