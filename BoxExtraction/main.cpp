#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/imgproc/imgproc.hpp>   
#include <opencv2/core/core.hpp>
#include <filesystem>
#include <regex>
#include <unordered_map>
#include <vector>

using namespace cv;
namespace fs = std::experimental::filesystem;
using marray = std::vector<std::vector<double>>;

int imgNum = 0;
int boxNum = 0;
int testNum = 0;
int trainNum = 0;
string rootServer = "/home/crb/datasets/wajueji/images/";

void SeparateImg(fs::path rootFolder) {
	if (!fs::exists(rootFolder) || !fs::is_directory(rootFolder))
	{
		throw std::runtime_error("Not a folder");
	}

	std::wregex sPattern(L"^(.*)(_s)$");

	fs::path rawSave = rootFolder / L"Raw";
	fs::path sSave = rootFolder / L"Raw_s";

	fs::create_directories(rawSave);
	fs::create_directories(sSave);

	using hash_map = std::unordered_map<std::wstring, std::wstring>;
	hash_map mapper;

	for (auto&fe : fs::directory_iterator(rootFolder))
	{
		fs::path fp = fe.path();
		if (fs::is_regular_file(fp))
		{
			std::wstring fn = fp.stem();

			std::wsmatch res;
			if (std::regex_match(fn, res, sPattern))
			{
				std::wstring rawName = res[1];

				fs::path rawImg = rootFolder / (rawName + L".jpg");

				fs::copy_file(rawImg, rawSave / rawImg.filename());
				fs::copy_file(fp, sSave / fp.filename());
			}
		}
	}
}

marray FindLabel(cv::Mat image) {
	marray res;
	int rows = image.rows;
	int cols = image.cols;
	Mat box(rows, cols, CV_8UC(1), Scalar::all(0));
	int label = 1;
	std::vector<int> maxRow;
	std::vector<int> maxCol;
	std::vector<int> minRow;
	std::vector<int> minCol;

	for (int i = 1; i < rows; ++i) {
		uchar b, g, r;
		for (int j = 1; j < cols; ++j) {
			b = image.at<cv::Vec3b>(i, j)[0];
			g = image.at<cv::Vec3b>(i, j)[1];
			r = image.at<cv::Vec3b>(i, j)[2];
			if (r < 245 || b > 10 && g > 10)
				continue;
			if (box.at<uchar>(i - 1, j) == 0 && box.at<uchar>(i, j - 1) == 0) {
				box.at<uchar>(i, j) = label;
				label++;
				maxRow.push_back(i);
				minRow.push_back(i);
				maxCol.push_back(j);
				minCol.push_back(j);
				continue;
			}
			if (box.at<uchar>(i - 1, j) == 0) {
				box.at<uchar>(i, j) = box.at<uchar>(i, j - 1);
			}
			else if (box.at<uchar>(i, j - 1) == 0) {
				box.at<uchar>(i, j) = box.at<uchar>(i - 1, j);
			}
			else {
				box.at<uchar>(i, j) = box.at<uchar>(i, j - 1) > box.at<uchar>(i - 1, j) ? box.at<uchar>(i - 1, j) : box.at<uchar>(i, j - 1);
			}
			int t = box.at<uchar>(i, j);
			if (i > maxRow[t - 1]) maxRow[t - 1] = i;
			if (i < minRow[t - 1]) minRow[t - 1] = i;
			if (j > maxCol[t - 1]) maxCol[t - 1] = j;
			if (j < minCol[t - 1]) minCol[t - 1] = j;
		}
	}
	for (int i = 0; i < label - 1; ++i)
	{
		if (maxRow[i] - minRow[i] < 5 || maxCol[i] - minCol[i] < 5) continue;
		std::vector<double> bounding(4);
		bounding[0] = (minCol[i] + maxCol[i]) / 2.0 / cols;
		bounding[1] = (minRow[i] + maxRow[i]) / 2.0 / rows;
		bounding[2] = (maxCol[i] - minCol[i]) / (double)cols;
		bounding[3] = (maxRow[i] - minRow[i]) / (double)rows;

		res.push_back(bounding);
	}

	return res;
}

string intToStrLen5(int i)
{
	std::stringstream ss;
	string s;
	char t[256];
	sprintf_s(t, "%05d", i);
	s = t;
	return s;
}

int wmain(int argc, wchar_t*argv[])
{
	fs::path rootFolder = argc == 1 ? L"./" : argv[1];
	if (!fs::exists(rootFolder) || !fs::is_directory(rootFolder))
	{
		throw std::runtime_error("Not a folder");
	}
	std::cout << rootFolder.string() << std::endl;

	fs::path imgSave = rootFolder / L"images";
	fs::path txtSave = rootFolder / L"labels";

	fs::create_directories(imgSave);
	fs::create_directories(txtSave);

	std::wregex sPattern(L"^(.*)(_s)$");

	std::ofstream ftrain(rootFolder.string() + "/train.txt");
	std::ofstream ftest(rootFolder.string() + "/test.txt");

	for (auto&fe : fs::directory_iterator(rootFolder))
	{
		fs::path fp = fe.path();
		if (fs::is_regular_file(fp))
		{
			std::wstring fn = fp.stem();

			std::wsmatch res;
			if (std::regex_match(fn, res, sPattern))
			{
				std::wstring rawName = res[1];
				fs::path rawImg = rootFolder / (rawName + L".jpg");

				string imgName = fp.string();
				//std::cout << imgName << std::endl;

				Mat img = imread(imgName);
				if (!img.data) {
					throw std::runtime_error("No img file");
				}

				imgNum++;
				string name = intToStrLen5(imgNum);

				std::ofstream fout(txtSave.string() + "/" + name + ".txt");

				marray labels;
				labels = FindLabel(img);
				if (labels.size() == 0) {
					throw std::runtime_error("No box");
				}
				for (auto &box : labels) {
					fout << 0;
					for (auto &it : box) {
						fout << ' ' << it;
					}
					fout << '\n';
					boxNum++;
				}

				if (testNum < 100 && imgNum % 3 == 0) {
					testNum++;
					ftest << (rootServer + name + ".jpg\n");
				}
				else {
					trainNum++;
					ftrain << (rootServer + name + ".jpg\n");
				}

				fs::copy_file(rawImg, imgSave / (name + ".jpg"));

				std::cout << "当前第" << imgNum << "张图片，共计" << boxNum << "个目标" << std::endl;

			}
		}
	}
	std::cout << "运行结束，训练集共" << trainNum << "张图片，测试集共" << testNum << "张图片" << std::endl;
	std::cout << "按Enter退出";
	getchar();

	return 0;
}