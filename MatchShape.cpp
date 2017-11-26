/**
 * @file MatchTemplate_Demo.cpp
 * @brief Sample code to use the function MatchTemplate
 * @author OpenCV team
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int number_count = 1;

double GetAreaPixelWeight(Mat image,vector<Point> area,int color_threshold,double number_threshold)
{
  int count = 0;
  //get the box of the area
  Rect bound_rect = boundingRect(area);
  //get all the point value in the area
  for(int i=bound_rect.x;i<bound_rect.width+bound_rect.x;i++)
  {
    for(int j=bound_rect.y;j<bound_rect.height+bound_rect.y;j++)
    {
      if( (image.at<Vec3b>(j,i)[2]<color_threshold) )
      {
        count++;
      }
    }
  }
  //cout<<(count*1.0/(bound_rect.width*bound_rect.height))<<endl;
  if( (count*1.0/(bound_rect.width*bound_rect.height))>number_threshold )
  {
    return true;
  }
  else
  {
    return false;
  }
}

int findTemple( string origin_image_path, string template_image_path,string out_name )
{

  //[load_image]
  Mat inputImg = imread(origin_image_path, CV_LOAD_IMAGE_COLOR);  
  Mat templateImg = imread(template_image_path, CV_LOAD_IMAGE_COLOR);

  if(inputImg.empty() || templateImg.empty())
  {
    cout << "Can't read one of the images" << endl;
    return -1;
  }

  //1.查找模版图像的轮廓  
  Mat copyImg1 = templateImg.clone();  
  cvtColor(templateImg, templateImg, CV_BGR2GRAY);  
  threshold(templateImg, templateImg, 100, 255, CV_THRESH_BINARY_INV);//确保黑中找白
  //imwrite("threshold.jpg",templateImg);    
  vector<vector<Point> > contours1;  
  findContours(templateImg, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);//最外层轮廓  
  drawContours(copyImg1, contours1, -1, Scalar(0, 255, 0), 2, 8);
  //imwrite("temple.jpg",copyImg1);  

  //2.查找待测试图像的轮廓   
  //convert to HSV
  Mat copyImg2 = inputImg.clone();
  vector<Mat> channels_rgb;
  split(copyImg2,channels_rgb);
  equalizeHist(channels_rgb[0],channels_rgb[0]);
  equalizeHist(channels_rgb[1],channels_rgb[1]);
  equalizeHist(channels_rgb[2],channels_rgb[2]);
  merge(channels_rgb,copyImg2);
  Mat copyImg3 = inputImg.clone();
  cvtColor(copyImg2,copyImg3,CV_BGR2HSV);
  vector<Mat> channels;
  split(copyImg3,channels);
  // imwrite("h_gamma.jpg", channels[0]); 
  // imwrite("s_gamma.jpg", channels[1]); 
  // imwrite("v_gamma.jpg", channels[2]); 
  
  Mat canny_image;
  Mat channels_value = channels[2].clone();
  equalizeHist(channels[2],canny_image);
  int threshold_value = threshold(channels[2], channels[2], 0, 255, CV_THRESH_BINARY_INV|CV_THRESH_OTSU);//确保黑中找白
  Canny(canny_image,canny_image,threshold_value,threshold_value*3);
  //imwrite("canny_"+out_name+".jpg",canny_image);
  Mat element = getStructuringElement( MORPH_RECT,Size(2,2));
  dilate( canny_image, canny_image, element );
  //imwrite("canny_add"+out_name+".jpg",canny_image);
  vector<vector<Point> > contours2;  
  findContours(canny_image, contours2, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);//最外层轮廓  
  
  Mat shape = inputImg.clone();

  
  //3.形状匹配---比较两个形状或轮廓间的相似度 
  for (int i = 0; i < contours2.size();i++)//遍历待测试图像的轮廓  
  {  
      Rect bound_rect = boundingRect(contours2[i]);
      if(bound_rect.width>20 && bound_rect.height>20)
      {
        double a0 = matchShapes(contours1[0],contours2[i],CV_CONTOURS_MATCH_I1,0);  
        if (a0<0.2 && GetAreaPixelWeight(copyImg3,contours2[i],50,0.3) )
        {        
          stringstream ss;
          ss<<number_count; 
          string s2 = ss.str();
          
          Mat train_image = canny_image(bound_rect);
          for(int x=0;x<train_image.cols;x++)
          {
            for(int y=0;y<train_image.rows;y++)
            {
              if(pointPolygonTest(contours2[i],Point2f(x+bound_rect.x,y+bound_rect.y),false)<0)
              {
                 train_image.at<char>(y,x)=0;
              }
            }
          }     
          //hough line
          Mat out_image = inputImg(bound_rect);
          vector<Vec2f> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
          HoughLines(train_image, lines, 1, CV_PI / 180, min(train_image.cols,train_image.rows)*0.4);
          //merge some lines
          for( size_t k = 0;k < lines.size();k++ )
          {
              float theta = lines[k][1];
              if(theta>CV_PI)
              {
                lines[k][0] = -lines[k][0];
                lines[k][1] = lines[k][1]-CV_PI;
              }
          }
          vector<Vec2f> merged_lines;
          for( size_t k = 0;k < lines.size();k++ )
          {
              if(merged_lines.size()==0)
              {
                merged_lines.push_back(lines[k]);
                continue;
              }
              float cur_rho = lines[k][0], cur_theta = lines[k][1];
              bool add_or_not = true;
              for(size_t m=0;m<merged_lines.size();m++)
              {
                float last_rho = merged_lines[m][0], last_theta = merged_lines[m][1];
                if(abs(cur_rho-last_rho)<4 && abs(cur_theta-last_theta)<0.1)
                {
                  merged_lines[m][0] = (cur_rho+last_rho)/2;
                  merged_lines[m][1] = (cur_theta+last_theta)/2;
                  add_or_not = false;
                  break;
                }
              }

              if(add_or_not)
              {
                merged_lines.push_back(lines[k]);
              }
          }

          for( size_t k = 0;k < merged_lines.size();k++ )//将求得的线条画出来
          {
              float rho = merged_lines[k][0], theta = merged_lines[k][1];
              Point pt1, pt2;
              double a = cos(theta), b = sin(theta);
              //cout<<"line: "<<rho<<" "<<theta<<endl;
              double x0 = a*rho, y0 = b*rho;
              pt1.x = cvRound(x0 + 1000*(-b));
              pt1.y = cvRound(y0 + 1000*(a));
              pt2.x = cvRound(x0 - 1000*(-b));
              pt2.y = cvRound(y0 - 1000*(a));
              line( out_image, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
          }

          //cout<<"Line size: "<<merged_lines.size()<<endl;
         
          //drawContours(shape, contours2, i, Scalar(0, 255, 0), 2, 8);//则在待测试图像上画出此轮廓 
          if(merged_lines.size()==3)  
          {
            Scalar     mean;  
            Scalar     stddev;  
            //cout << "模版轮廓与待测试图像轮廓" << i << "的相似度:" << a0 << endl;//输出两个轮廓间的相似度 
            Mat mask_image = channels[2](bound_rect);
            for(int x=0;x<train_image.cols;x++)
            {
              for(int y=0;y<train_image.rows;y++)
              {
                if(pointPolygonTest(contours2[i],Point2f(x+bound_rect.x,y+bound_rect.y),false)<0)
                {
                  mask_image.at<char>(y,x)=char(0);
                }
                else
                {
                  mask_image.at<char>(y,x)=char(255);
                }
              }
            } 
            Mat calc_image = channels_value(bound_rect);
            equalizeHist(calc_image,calc_image);
            cv::meanStdDev ( calc_image, mean, stddev,mask_image);
            cout<<s2<<"\t"<<mean.val[0]<<"\t"<<stddev.val[0]<<endl;//<<"\t"<<mean.val[2]<<"\t"<<stddev.val[0]<<"\t"<<stddev.val[1]<<"\t"<<stddev.val[2]<<endl;
            imwrite("Train/Train_"+s2+"_mask.jpg",mask_image);
            imwrite("Train/Train_"+s2+".jpg",out_image);
            rectangle(shape,bound_rect,Scalar(0,0,255),2,8); 
            number_count++;
          }
        } 
         
      }
  }
  imwrite(out_name+".jpg",shape);

  return 0;
}


int main()
{
  vector<string> image_name;
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142414.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142418.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142423.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142427.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142431.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142435.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142440.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142444.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142448.jpg");
  image_name.push_back("/home/yan/Desktop/20170908142409/mob20170908142452.jpg");

  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150434.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150438.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150442.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150447.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150450.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150454.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150458.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150502.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150507.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150510.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150514.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150517.jpg");
  image_name.push_back("/home/yan/Desktop/20170908150430/mob20170908150520.jpg");
  
  for(int i=0;i<image_name.size();i++)
  {
    stringstream ss;
    ss<<i; 
    string s1 = ss.str();
    findTemple(image_name[i],"/home/yan/Desktop/20170908142409/0.jpg",s1);
  }
  
}