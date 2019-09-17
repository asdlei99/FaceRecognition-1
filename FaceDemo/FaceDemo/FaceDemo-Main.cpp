/*
����ʶ��Demo
*/

//C++ͷ�ļ�
#include <iostream>
#include <vector>
#include <fstream>

//Opencvͷ�ļ�
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"

//���ļ�
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

//����ѧϰ
bool LearningFace(cv::Ptr<cv::face::LBPHFaceRecognizer>* pModel, int* pWidth, int* pHeight);

//�������
bool AnaliseFace(cv::Ptr<cv::face::LBPHFaceRecognizer>* pModel, int* pWidth, int* pHeight);

int main(int argc, char* argv[], char* enp[])
{
	//�����Ŀ�Ⱥ͸߶�
	int nFaceWidth = 0, nFaceHeight = 0;

	//ѧϰģ��
	cv::Ptr<cv::face::LBPHFaceRecognizer> cModel = cv::face::LBPHFaceRecognizer::create();

	//����ѧϰ
	std::cout << "����ѧϰ��ʼ...." << std::endl;
	LearningFace(&cModel, &nFaceWidth, &nFaceHeight);
	std::cout << "����ѧϰ����...." << std::endl;

	//�������
	std::cout << "������⿪ʼ...." << std::endl;
	AnaliseFace(&cModel, &nFaceWidth, &nFaceHeight);
	std::cout << "����������...." << std::endl;

	getchar();
	return 0;
}

bool LearningFace(cv::Ptr<cv::face::LBPHFaceRecognizer>* pModel, int* pWidth, int* pHeight)
{
	try
	{
		//�������ļ�
		std::ifstream cFile("H:/FirstData.txt", std::fstream::in);
		if (!cFile.is_open())
			throw std::runtime_error("���ļ�ʧ��");

		//��������ͼ��·���Ͷ�Ӧ�����
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
			throw std::runtime_error("û��ͼ�����ѧϰ");

		//��ȡ�����Ŀ�Ⱥ͸߶�
		*pWidth = cImage[0].cols;
		*pHeight = cImage[0].rows;

		//����ѧϰ
		(*pModel)->train(cImage, cLable);

		//��ѧϰ��ɵ����ݽ��б���
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
		//������ͷ
		cv::VideoCapture cCapture(0);
		if (!cCapture.isOpened())
			throw std::runtime_error("������ͷʧ��");

		//��ȡ����ͷ�����ͼ��Ŀ�Ⱥ͸߶�
		cv::Size cFrameSize = cv::Size(static_cast<int>(cCapture.get(cv::CAP_PROP_FRAME_WIDTH)),
			static_cast<int>(cCapture.get(cv::CAP_PROP_FRAME_HEIGHT)));

		//����һ������������ʾͼ��
		cv::namedWindow("FaceDemo",cv::WINDOW_AUTOSIZE);

		//������⼶�����ݿ�
		std::string cHaarFace = "haarcascade_frontalface_alt_tree.xml";

		//����һ��������⼶��
		cv::CascadeClassifier cFaceCascade;
		cFaceCascade.load(cHaarFace);

		//����һ֡������ͷͼ��
		cv::Mat cFrame;

		//�ҶȺ����������
		cv::Mat cDest;

		//��С�任�����������
		cv::Mat cFinishDest;

		//������������
		std::vector<cv::Rect> cFaceRectArray;

		//��ʼѭ����ȡ����ͷ��ͼ����Ϣ
		while (cCapture.read(cFrame))
		{
			//����ͼ��ľ���ģʽ
			cv::flip(cFrame, cFrame, 1);

			//��ͼ���е��������м��
			cFaceCascade.detectMultiScale(cFrame, cFaceRectArray, 1.1, 3, 0/*, cv::Size(50, 50), cv::Size(500, 500)*/);

			//��ÿһ���������в���
			for (int i = 0; i < cFaceRectArray.size(); i++)
			{
				//����������Դ
				static int nIndex = 0;
				if (cv::waitKey(1) == 'a')
				{
					cv::Mat cDest;
					cv::resize(cFrame(cFaceRectArray[i]), cDest, cv::Size(100, 100));
					cv::imwrite(cv::format("H:/Me_4/Face_%d.png", nIndex++), cDest);
					std::cout << nIndex << std::endl;
				}

				//��������Ϊ�Ҷ�ͼ
				cv::cvtColor(cFrame(cFaceRectArray[i]), cDest, cv::COLOR_RGB2GRAY);

				//ת��Ϊָ���Ĵ�С
				cv::resize(cDest, cFinishDest, cv::Size(*pWidth, *pHeight));

				//�����ж�
				int nFaceIndex = (*pModel)->predict(cFinishDest);
				if(nFaceIndex == 1)
					std::cout << "���۾�" << nFaceIndex << std::endl;
				else if(nFaceIndex == 2)
					std::cout << "û�۾�" << nFaceIndex << std::endl;
				else if(nFaceIndex == 3)
					std::cout << "С����" << nFaceIndex << std::endl;
				else if(nFaceIndex == 4)
					std::cout << "Сȫȫ" << nFaceIndex << std::endl;

				//��ÿһ������Ȧ����
				cv::rectangle(cFrame, cFaceRectArray[i], cv::Scalar(0, 0, 255), 2, 8, 0);
			}

			//��ʾ��ȡ�õ���ͼ��
			cv::imshow("FaceDemo", cFrame);

			//����q�˳�ѭ��
			if(cv::waitKey(1) == 27)break;
		}

		//�ͷ��������Դ
		cCapture.release();

		//�ͷŴ�����Դ
		cv::destroyWindow("FaceDemo");

	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}
	return true;
}
