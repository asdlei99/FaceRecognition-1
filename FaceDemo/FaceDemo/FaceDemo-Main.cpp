/*
人脸识别Demo
*/

//C++头文件
#include <iostream>
#include <vector>
#include <fstream>

//Opencv头文件
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"

//库文件
#pragma comment(lib,"opencv_aruco411d.lib")
#pragma comment(lib,"opencv_bgsegm411d.lib")
#pragma comment(lib,"opencv_bioinspired411d.lib")
#pragma comment(lib,"opencv_calib3d411d.lib")
#pragma comment(lib,"opencv_ccalib411d.lib")
#pragma comment(lib,"opencv_core411d.lib")
#pragma comment(lib,"opencv_datasets411d")
#pragma comment(lib,"opencv_dnn_objdetect411d")
#pragma comment(lib,"opencv_dnn411d")
#pragma comment(lib,"opencv_dpm411d")
#pragma comment(lib,"opencv_face411d")
#pragma comment(lib,"opencv_features2d411d")
#pragma comment(lib,"opencv_flann411d")
#pragma comment(lib,"opencv_fuzzy411d")
#pragma comment(lib,"opencv_gapi411d")
#pragma comment(lib,"opencv_hfs411d")
#pragma comment(lib,"opencv_highgui411d")
#pragma comment(lib,"opencv_img_hash411d")
#pragma comment(lib,"opencv_imgcodecs411d")
#pragma comment(lib,"opencv_imgproc411d")
#pragma comment(lib,"opencv_line_descriptor411d")
#pragma comment(lib,"opencv_ml411d")
#pragma comment(lib,"opencv_objdetect411d")
#pragma comment(lib,"opencv_optflow411d")
#pragma comment(lib,"opencv_phase_unwrapping411d")
#pragma comment(lib,"opencv_photo411d")
#pragma comment(lib,"opencv_plot411d")
#pragma comment(lib,"opencv_quality411d")
#pragma comment(lib,"opencv_reg411d")
#pragma comment(lib,"opencv_rgbd411d")
#pragma comment(lib,"opencv_saliency411d")
#pragma comment(lib,"opencv_shape411d")
#pragma comment(lib,"opencv_stereo411d")
#pragma comment(lib,"opencv_stitching411d")
#pragma comment(lib,"opencv_structured_light411d")
#pragma comment(lib,"opencv_superres411d")
#pragma comment(lib,"opencv_surface_matching411d")
#pragma comment(lib,"opencv_text411d")
#pragma comment(lib,"opencv_tracking411d")
#pragma comment(lib,"opencv_video411d")
#pragma comment(lib,"opencv_videoio411d")
#pragma comment(lib,"opencv_videostab411d")
#pragma comment(lib,"opencv_xfeatures2d411d")
#pragma comment(lib,"opencv_ximgproc411d")
#pragma comment(lib,"opencv_xobjdetect411d")
#pragma comment(lib,"opencv_xphoto411d")

//人脸学习
bool LearningFace(cv::Ptr<cv::face::LBPHFaceRecognizer>* pModel, int* pWidth, int* pHeight);

//人脸检测
bool AnaliseFace(cv::Ptr<cv::face::LBPHFaceRecognizer>* pModel, int* pWidth, int* pHeight);

int main(int argc, char* argv[], char* enp[])
{
	//人脸的宽度和高度
	int nFaceWidth = 0, nFaceHeight = 0;

	//学习模型
	cv::Ptr<cv::face::LBPHFaceRecognizer> cModel = cv::face::LBPHFaceRecognizer::create();

	//人脸学习
	std::cout << "人脸学习开始...." << std::endl;
	LearningFace(&cModel, &nFaceWidth, &nFaceHeight);
	std::cout << "人脸学习结束...." << std::endl;

	//人脸检测
	std::cout << "人脸检测开始...." << std::endl;
	AnaliseFace(&cModel, &nFaceWidth, &nFaceHeight);
	std::cout << "人脸检测结束...." << std::endl;

	getchar();
	return 0;
}

bool LearningFace(cv::Ptr<cv::face::LBPHFaceRecognizer>* pModel, int* pWidth, int* pHeight)
{
	try
	{
		//打开配置文件
		std::ifstream cFile("H:/FirstData.txt", std::fstream::in);
		if (!cFile.is_open())
			throw std::runtime_error("打开文件失败");

		//解析出来图像路径和对应的序号
		std::string strLine, strPath, strLable;
		std::vector<cv::Mat> cImage;
		std::vector<int> cLable;
		char cBetween = ';';
		while (std::getline(cFile, strLine))
		{
			std::stringstream cStreamLine(strLine);
			std::getline(cStreamLine, strPath, cBetween);
			std::getline(cStreamLine, strLable);
			if (strPath.size() && strLable.size())
			{
				cv::Mat cFace = cv::imread(strPath, 0);
				if (cFace.rows && cFace.cols)
				{
					cImage.emplace_back(cFace);
					cLable.emplace_back(std::atoi(strLable.c_str()));
				}
			}
		}

		if (!cImage.size() || !cLable.size())
			throw std::runtime_error("没有图像进行学习");

		//获取人脸的宽度和高度
		*pWidth = cImage[0].cols;
		*pHeight = cImage[0].rows;

		//人脸学习
		(*pModel)->train(cImage, cLable);

		//将学习完成的内容进行保存
		(*pModel)->save("H:/TrainData.xml");

	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}
	return true;
}

bool AnaliseFace(cv::Ptr<cv::face::LBPHFaceRecognizer>* pModel, int* pWidth, int* pHeight)
{
	try
	{
		//打开摄像头
		cv::VideoCapture cCapture(0);
		if (!cCapture.isOpened())
			throw std::runtime_error("打开摄像头失败");

		//获取摄像头捕获的图像的宽度和高度
		cv::Size cFrameSize = cv::Size(static_cast<int>(cCapture.get(cv::CAP_PROP_FRAME_WIDTH)),
			static_cast<int>(cCapture.get(cv::CAP_PROP_FRAME_HEIGHT)));

		//创建一个窗口用来显示图像
		cv::namedWindow("FaceDemo",cv::WINDOW_AUTOSIZE);

		//人脸检测级联数据库
		std::string cHaarFace = "haarcascade_frontalface_alt_tree.xml";

		//创建一个人脸检测级联
		cv::CascadeClassifier cFaceCascade;
		cFaceCascade.load(cHaarFace);

		//保存一帧的摄像头图像
		cv::Mat cFrame;

		//灰度后的人脸数据
		cv::Mat cDest;

		//大小变换后的人脸数据
		cv::Mat cFinishDest;

		//人脸区域容器
		std::vector<cv::Rect> cFaceRectArray;

		//开始循环获取摄像头的图像信息
		while (cCapture.read(cFrame))
		{
			//消除图像的镜像模式
			cv::flip(cFrame, cFrame, 1);

			//对图像中的人脸进行检测
			cFaceCascade.detectMultiScale(cFrame, cFaceRectArray, 1.1, 3, 0/*, cv::Size(50, 50), cv::Size(500, 500)*/);

			//对每一张人脸进行操作
			for (int i = 0; i < cFaceRectArray.size(); i++)
			{
				//保存人脸资源
				static int nIndex = 0;
				if (cv::waitKey(1) == 'a')
				{
					cv::Mat cDest;
					cv::resize(cFrame(cFaceRectArray[i]), cDest, cv::Size(100, 100));
					cv::imwrite(cv::format("H:/Me_4/Face_%d.png", nIndex++), cDest);
					std::cout << nIndex << std::endl;
				}

				//处理人脸为灰度图
				cv::cvtColor(cFrame(cFaceRectArray[i]), cDest, cv::COLOR_RGB2GRAY);

				//转化为指定的大小
				cv::resize(cDest, cFinishDest, cv::Size(*pWidth, *pHeight));

				//人脸判断
				int nFaceIndex = (*pModel)->predict(cFinishDest);
				if(nFaceIndex == 1)
					std::cout << "戴眼镜" << nFaceIndex << std::endl;
				else if(nFaceIndex == 2)
					std::cout << "没眼镜" << nFaceIndex << std::endl;
				else if(nFaceIndex == 3)
					std::cout << "小威威" << nFaceIndex << std::endl;
				else if(nFaceIndex == 4)
					std::cout << "小全全" << nFaceIndex << std::endl;

				//将每一张人脸圈起来
				cv::rectangle(cFrame, cFaceRectArray[i], cv::Scalar(0, 0, 255), 2, 8, 0);
			}

			//显示获取得到的图像
			cv::imshow("FaceDemo", cFrame);

			//按键q退出循环
			if(cv::waitKey(1) == 27)break;
		}

		//释放摄像机资源
		cCapture.release();

		//释放窗口资源
		cv::destroyWindow("FaceDemo");

	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}
	return true;
}
