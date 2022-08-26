#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

void WDT(Mat& src, Mat& dst)
{ 
	Mat highfilter(1, 2, CV_32FC1);
	Mat lowfilter(1, 2, CV_32FC1);
	highfilter.at <float>(0, 0) = -1 / sqrtf(2);
	highfilter.at <float>(0, 1) =  1 / sqrtf(2);
	lowfilter.at <float>(0, 0) = 1 / sqrtf(2);
	lowfilter.at <float>(0, 1) = 1 / sqrtf(2);

	int row = src.rows;
	int col = src.cols;


	//行小波变换
	for (int i = 0; i < row; i++)
	{
		Mat oneRow(1, col, CV_32FC1);
		for (int j = 0; j < col; j++)
		{
			oneRow.at<float>(0, j) = src.at<uchar>(i, j);
		}
		//逐行滤波
		Mat dst1(1, col, CV_32FC1); //低通分量
		Mat dst2(1, col, CV_32FC1); //高通分量

		for (int k = 1; k < col; k++)
		{
			dst1.at<float>(0, k) = oneRow.at<float>(0, k - 1) * lowfilter.at<float>(0, 0) + oneRow.at<float>(0, k) * lowfilter.at<float>(0, 1);
			
		}


		for (int k = 1; k < col; k++)
		{
			dst2.at<float>(0, k) = oneRow.at<float>(0, k - 1) * highfilter.at<float>(0, 0) + oneRow.at<float>(0, k) * highfilter.at<float>(0, 1);
		}
		//低通高通拼接
		for (int p = 0, q = 1; p < col / 2; p++, q += 2)
		{
			oneRow.at<float>(0, p) = dst1.at<float>(0, q);
			oneRow.at<float>(0, p + col / 2) = dst2.at<float>(0, q);
		}

		for (int j = 0; j < col; j++)
		{
			dst.at<float>(i, j) = oneRow.at<float>(0, j);
		}
	}

	//列小波变换
	for (int j = 0; j < col; j++)
	{
		Mat oneCol(1, row, CV_32FC1);
		for (int i = 0; i < row; i++)
		{
			oneCol.at<float>(0, i) = dst.at<float>(i, j);
		}
		//逐行滤波
		Mat dst3(1, row, CV_32FC1); //低通分量
		Mat dst4(1, row, CV_32FC1); //高通分量

		for (int k = 1; k < row; k++)
		{
			dst3.at<float>(0, k) = oneCol.at<float>(0, k - 1) * lowfilter.at<float>(0, 0) + oneCol.at<float>(0, k) * lowfilter.at<float>(0, 1);

		}


		for (int k = 1; k < row; k++)
		{
			dst4.at<float>(0, k) = oneCol.at<float>(0, k - 1) * highfilter.at<float>(0, 0) + oneCol.at<float>(0, k) * highfilter.at<float>(0, 1);
		}

		//低通高通拼接
		for (int p = 0, q = 1; p < row / 2; p++, q += 2)
		{
			oneCol.at<float>(0, p) = dst3.at<float>(0, q);
			oneCol.at<float>(0, p + row / 2) = dst4.at<float>(0, q);
		}

		for (int i = 0; i < row; i++)
		{
			dst.at<float>(i, j) = oneCol.at<float>(0, i);
		}
	}

}

void IWDT(Mat& src, Mat& ini, Mat& dst)
{
	//将src的左上角替换为ini
	for (int i = 0; i < ini.rows; i++)
	{
		for (int j = 0; j < ini.cols; j++)
		{
			src.at<float>(i, j) = ini.at<uchar>(i, j) * 2;
		}
	}

	Mat highfilter(1, 2, CV_32FC1);
	Mat lowfilter(1, 2, CV_32FC1);
	highfilter.at <float>(0, 0) = 1 / sqrtf(2);
	highfilter.at <float>(0, 1) = -1 / sqrtf(2);
	lowfilter.at <float>(0, 0) = 1 / sqrtf(2);
	lowfilter.at <float>(0, 1) = 1 / sqrtf(2);

	int row = src.rows;
	int col = src.cols;
	//列逆变换
	for (int j = 0; j < col; j++)
	{
		Mat oneCol(1, row, CV_32FC1);
		for (int i = 0; i < row; i++)
		{
			oneCol.at<float>(0, i) = src.at<float>(i, j);
		}

		//重建
		Mat up1(Mat::zeros(1, row, CV_32FC1));
		Mat up2(Mat::zeros(1, row, CV_32FC1));
		for (int i = 0, cnt = 0; i < row / 2; i++, cnt += 2)
		{
			up1.at<float>(0, cnt) = oneCol.at<float>(0, i);
			up2.at<float>(0, cnt) = oneCol.at<float>(0, i + row/2);
		}

		//卷积
		Mat dst1(Mat::zeros(1, row, CV_32FC1));
		Mat dst2(Mat::zeros(1, row, CV_32FC1));
		for (int k = 1; k < row; k++)
		{
			dst1.at<float>(0, k) = up1.at<float>(0, k - 1) * lowfilter.at<float>(0, 0) + up1.at<float>(0, k) * lowfilter.at<float>(0, 1);

		}


		for (int k = 1; k < row; k++)
		{
			dst2.at<float>(0, k) = up2.at<float>(0, k - 1) * highfilter.at<float>(0, 0) + up2.at<float>(0, k) * highfilter.at<float>(0, 1);
		}

		//加起来
		for (int k = 1; k < row; k++)
		{
			oneCol.at<float>(0, k) = dst1.at<float>(0, k) + dst2.at<float>(0, k);
		}

		for (int i = 0; i < row; i++)
		{
			dst.at<float>(i, j) = oneCol.at<float>(0, i);
		}
	}

	//行逆变换
	for (int i = 0; i < row; i++)
	{
		Mat oneRow(1, col, CV_32FC1);
		for (int j = 0; j < col; j++)
		{
			oneRow.at<float>(0, j) = dst.at<float>(i, j);
		}

		//重建
		Mat up1(Mat::zeros(1, col, CV_32FC1));
		Mat up2(Mat::zeros(1, col, CV_32FC1));
		for (int j = 0, cnt = 0; j < col / 2; j++, cnt += 2)
		{
			up1.at<float>(0, cnt) = oneRow.at<float>(0, j);
			up2.at<float>(0, cnt) = oneRow.at<float>(0, j + col / 2);
		}

		//卷积
		Mat dst1(Mat::zeros(1, col, CV_32FC1));
		Mat dst2(Mat::zeros(1, col, CV_32FC1));
		for (int k = 1; k < col; k++)
		{
			dst1.at<float>(0, k) = up1.at<float>(0, k - 1) * lowfilter.at<float>(0, 0) + up1.at<float>(0, k) * lowfilter.at<float>(0, 1);

		}


		for (int k = 1; k < col; k++)
		{
			dst2.at<float>(0, k) = up2.at<float>(0, k - 1) * highfilter.at<float>(0, 0) + up2.at<float>(0, k) * highfilter.at<float>(0, 1);
		}

		//加起来
		for (int k = 1; k < col; k++)
		{
			oneRow.at<float>(0, k) = dst1.at<float>(0, k) + dst2.at<float>(0, k);
		}

		for (int j = 0; j < col; j++)
		{
			dst.at<float>(i, j) = oneRow.at<float>(0, j);
		}
	}

}


void float2uchar(Mat& dst, Mat& out)
{
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			out.at<uchar>(i, j) = (uchar)dst.at<float>(i, j);
		}
	}
}

int main(int arge ,char ** argv)
{
   //in存放未处理过的540*960图片像素   
   //颜色模式设置
	Mat in = imread("C:\\Users\\kris9\\Desktop\\JCS\\集创赛数据集\\downscaled\\41.bmp",0);  //enum::ImreadModes = 0 灰度模式
	//Mat in = imread("C:\\Users\\kris9\\Desktop\\JCS\\集创赛数据集\\downscaled\\41.bmp", 1);   //enum::ImreadModes = 1 3通道color模式

    //初始化src存放线性插值过的1080*1920图片像素
	Mat src;

	//线性插值放缩，放缩后存放在src中
	resize(in, src, Size(0, 0),2,2,INTER_LINEAR);
	//resize(in, src, Size(0, 0), 2, 2, INTER_NEAREST);

	//初始化dst，用来存放小波正变换处理后像素数据
	Mat dst(src.rows,src.cols,CV_32FC1);
	//初始化iwdt，用来存放小波逆变换处理后像素数据
	Mat dst_iwdt(src.rows, src.cols, CV_32FC1);
	
	//wavelet
	WDT(src, dst);
	//逆wavelet
	IWDT(dst, in, dst_iwdt);


	Mat out(dst.rows, dst.cols, CV_8UC1);

	float2uchar(dst_iwdt, out);
	imshow("output",out);
	waitKey(0);
	return 0;
}

