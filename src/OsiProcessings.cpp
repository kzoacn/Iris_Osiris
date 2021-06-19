/*******************************************************
* Open Source for Iris : OSIRIS
* Version : 4.0
* Date : 2011
* Author : Guillaume Sutra, Telecom SudParis, France
* License : BSD
********************************************************/

#include "opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#include "OsiStringUtils.h"
#include "OsiProcessings.h"


using namespace std ;

namespace osiris
{    

    OsiProcessings::OsiProcessings()
    {
        // Do nothing
    }


    OsiProcessings::~OsiProcessings()
    {
        // Do nothing
    }

    void OsiProcessings::segment ( const cv::Mat pSrc ,
                                         cv::Mat pMask ,
                                         OsiCircle & rPupil ,
                                         OsiCircle & rIris ,
										 vector<float> & rThetaCoarsePupil ,
										 vector<float> & rThetaCoarseIris ,
										 vector<cv::Point> & rCoarsePupilContour ,
										 vector<cv::Point> & rCoarseIrisContour ,
                                         int minIrisDiameter ,
                                         int minPupilDiameter ,
                                         int maxIrisDiameter ,
                                         int maxPupilDiameter )
    {

        // Check arguments
        //////////////////

        // String functions
        OsiStringUtils str ;

        // Temporary int to check sizes of pupil and iris
        int check_size = 0 ;

        // Default value for maxIrisDiameter if user did not specify it
        if ( maxIrisDiameter == 0 )
        {
            maxIrisDiameter = min(pSrc->height,pSrc->width) ;
        }

        // Change maxIrisDiameter if it is too big relative to image sizes
        else if ( maxIrisDiameter > (check_size = floor((float)min(pSrc->height,pSrc->width))) )
        {            
            cout << "Warning in function segment : maxIrisDiameter = " << maxIrisDiameter ;
            cout << " is replaced by " << check_size ;
            cout << " because image size is " << pSrc->width << "x" << pSrc->height << endl ;
            maxIrisDiameter = check_size ;
        }

        // Default value for maxPupilDiameter if user did not specify it
        if ( maxPupilDiameter == 0 )
        {
            maxPupilDiameter = OSI_MAX_RATIO_PUPIL_IRIS * maxIrisDiameter ;
        }
        
        // Change maxPupilDiameter if it is too big relative to maxIrisDiameter and OSI_MAX_RATIO_PUPIL_IRIS
        else if ( maxPupilDiameter > (check_size = OSI_MAX_RATIO_PUPIL_IRIS*maxIrisDiameter) )
        {
            cout << "Warning in function segment : maxPupilDiameter = " << maxPupilDiameter ;
            cout << " is replaced by " << check_size ;
            cout << " because maxIrisDiameter = " << maxIrisDiameter ;
            cout << " and ratio pupil/iris is generally lower than " << OSI_MAX_RATIO_PUPIL_IRIS << endl ;
            maxPupilDiameter = check_size ;
        }

        // Change minIrisDiameter if it is too small relative to OSI_SMALLEST_IRIS
        if ( minIrisDiameter < (check_size = OSI_SMALLEST_IRIS) )
        {
            cout << "Warning in function segment : minIrisDiameter = " << minIrisDiameter ;
            cout << " is replaced by " << check_size ;
            cout << " which is the smallest size for detecting iris" << endl ;
            minIrisDiameter = check_size ;
        }

        // Change minPupilDiameter if it is too small relative to minIrisDiameter and OSI_MIN_RATIO_PUPIL_IRIS
        if ( minPupilDiameter < (check_size = minIrisDiameter*OSI_MIN_RATIO_PUPIL_IRIS) )
        {
            cout << "Warning in function segment : minPupilDiameter = " << minPupilDiameter ;
            cout << " is replaced by " << check_size ;
            cout << " because minIrisDiameter = " << minIrisDiameter ;
            cout << " and ratio pupil/iris is generally upper than " << OSI_MIN_RATIO_PUPIL_IRIS << endl ;
            minIrisDiameter = check_size ;
        }

        // Check that minIrisDiameter < maxIrisDiameter
        if ( minIrisDiameter > maxIrisDiameter )
        {
            throw invalid_argument("Error in function segment : minIrisDiameter = " +
                                   str.toString(minIrisDiameter) +
                                   " should be lower than maxIrisDiameter = " +
                                   str.toString(maxIrisDiameter)) ;
        }

        // Make size odds
        minIrisDiameter += ( minIrisDiameter % 2 ) ? 0 : -1 ;
        maxIrisDiameter += ( maxIrisDiameter % 2 ) ? 0 : +1 ;
        minPupilDiameter += ( minPupilDiameter % 2 ) ? 0 : -1 ;
        maxPupilDiameter += ( maxPupilDiameter % 2 ) ? 0 : +1 ;





        // Start processing
        ///////////////////


        // Locate the pupil
        detectPupil(pSrc,rPupil,minPupilDiameter,maxPupilDiameter) ;

        // Fill the holes in an area surrounding pupil
        cv::Mat clone_src = pSrc.clone() ;
        cvSetImageROI(clone_src,cvRect(rPupil.getCenter().x-3.0/4.0*maxIrisDiameter/2.0,
                                       rPupil.getCenter().y-3.0/4.0*maxIrisDiameter/2.0,
                                       3.0/4.0*maxIrisDiameter,
                                       3.0/4.0*maxIrisDiameter)) ;
        fillWhiteHoles(clone_src,clone_src) ;
        cvResetImageROI(clone_src) ;

        // Will contain samples of angles, in radians
        vector<float> theta ;
        float theta_step = 0 ;



        // Pupil Accurate Contour
        /////////////////////////

        theta.clear() ;
        theta_step = 360.0 / OSI_PI / rPupil.getRadius() ;
        for ( float t = 0 ; t < 360 ; t += theta_step )
        {
            theta.push_back(t*OSI_PI/180) ;
        }
        vector<cv::Point> pupil_accurate_contour = findContour(clone_src,
                                                             rPupil.getCenter(),
                                                             theta,
                                                             rPupil.getRadius()-20,
                                                             rPupil.getRadius()+20) ;
        
        // Circle fitting on accurate contour
        rPupil.computeCircleFitting(pupil_accurate_contour) ;




        // Pupil Coarse Contour
        ///////////////////////

        theta.clear() ;
        theta_step = 360.0 / OSI_PI / rPupil.getRadius() * 2 ;
        for ( float t = 0 ; t < 360 ; t += theta_step )
        {
            if ( t > 45 && t < 135 ) t += theta_step ;
            theta.push_back(t*OSI_PI/180) ;
        }
        vector<cv::Point> pupil_coarse_contour = findContour(clone_src,
                                                           rPupil.getCenter(),
                                                           theta,
                                                           rPupil.getRadius()-20,
                                                           rPupil.getRadius()+20) ;

		rThetaCoarsePupil = theta ;
		rCoarsePupilContour = pupil_coarse_contour ;

        // Circle fitting on coarse contour
        rPupil.computeCircleFitting(pupil_coarse_contour) ;




        // Mask of pupil
        ////////////////

        cv::Mat mask_pupil = pSrc.clone() ;
        mask_pupil=0 ;
        drawContour(mask_pupil,pupil_accurate_contour,cv::Scalar(255),-1) ;

        


        // Iris Coarse Contour
        //////////////////////
                
        theta.clear() ;
        int min_radius = max<int>(rPupil.getRadius()/OSI_MAX_RATIO_PUPIL_IRIS,minIrisDiameter/2) ;
        int max_radius = min<int>(rPupil.getRadius()/OSI_MIN_RATIO_PUPIL_IRIS,3*maxIrisDiameter/4) ;
        theta_step = 360.0 / OSI_PI / min_radius ;
        for ( float t = 0 ; t < 360 ; t += theta_step )
        {
            if ( t < 180 || ( t > 225 && t < 315 ) ) t += 2*theta_step ;
            theta.push_back(t*OSI_PI/180) ;
        }        
        vector<cv::Point> iris_coarse_contour = findContour(clone_src,
                                                          rPupil.getCenter(),
                                                          theta,
                                                          min_radius,
                                                          max_radius) ;

		rThetaCoarseIris = theta ;
		rCoarseIrisContour = iris_coarse_contour ;

        // Circle fitting on coarse contour
        rIris.computeCircleFitting(iris_coarse_contour) ;

        // Mask of iris
        ///////////////

        cv::Mat mask_iris = mask_pupil.clone() ;
        mask_iris=0 ;
        drawContour(mask_iris,iris_coarse_contour,cv::Scalar(255),-1) ;




        // Iris Accurate Contour
        ////////////////////////
        
        // For iris accurate contour, limit the search of contour inside a mask
        // mask = dilate(mask-iris) - dilate(mask_pupil)

        // Dilate mask of iris by a disk-shape element
        cv::Mat mask_iris2 = mask_iris.clone() ;
        IplConvKernel * struct_element = cvCreateStructuringElementEx(21,21,10,10,CV_SHAPE_ELLIPSE) ;
        //cvMorphologyEx(mask_iris2,mask_iris2,mask_iris2,struct_element,CV_MOP_DILATE) ;
        cvDilate(mask_iris2,mask_iris2,struct_element) ;
        cvReleaseStructuringElement(&struct_element) ;

        // Dilate the mask of pupil by a horizontal line-shape element
        cv::Mat mask_pupil2 = mask_pupil.clone() ;
        struct_element = cvCreateStructuringElementEx(21,21,10,1,CV_SHAPE_RECT) ;
        //cvMorphologyEx(mask_pupil2,mask_pupil2,mask_pupil2,struct_element,CV_MOP_DILATE) ;
        cvDilate(mask_pupil2,mask_pupil2,struct_element) ;
        cvReleaseStructuringElement(&struct_element) ;

        // dilate(mask_iris) - dilate(mask_pupil)
        cvXor(mask_iris2,mask_pupil2,mask_iris2) ;
        
        theta.clear() ;
        theta_step = 360.0 / OSI_PI / rIris.getRadius() ;
        for ( float t = 0 ; t < 360 ; t += theta_step )
        {
            theta.push_back(t*OSI_PI/180) ;
        }        
        vector<cv::Point> iris_accurate_contour = findContour(clone_src,
                                                            rPupil.getCenter(),
                                                            theta,
                                                            rIris.getRadius()-50,
                                                            rIris.getRadius()+20,
                                                            mask_iris2) ;

        // Release memory
        
        





        // Mask of iris based on accurate contours
        //////////////////////////////////////////
        
        mask_iris=0 ;
        drawContour(mask_iris,iris_accurate_contour,cv::Scalar(255),-1) ;
        cvXor(mask_iris,mask_pupil,mask_iris) ;


        // Refine the mask by removing some noise
        /////////////////////////////////////////

        // Build a safe area = avoid occlusions
        cv::Mat safe_area = mask_iris.clone() ;
        cvRectangle(safe_area,cv::Point(0,0),cv::Point(safe_area->width-1,rPupil.getCenter().y),cv::Scalar(0),-1) ;
        cvRectangle(safe_area,cv::Point(0,rPupil.getCenter().y+rPupil.getRadius()),
                                      cv::Point(safe_area->width-1,safe_area->height-1),cv::Scalar(0),-1) ;
        struct_element = cvCreateStructuringElementEx(11,11,5,5,CV_SHAPE_ELLIPSE) ;
        //cvMorphologyEx(safe_area,safe_area,safe_area,struct_element,CV_MOP_ERODE) ;
        cvErode(safe_area,safe_area,struct_element) ;
        cvReleaseStructuringElement(&struct_element) ;

        // Compute the mean and the variance of iris texture inside safe area
        //double iris_mean = cvMean(pSrc,safe_area) ;
		cv::Scalar iris_mean = cvAvg(pSrc, safe_area);
        cv::Mat variance = cv::Mat(pSrc.size(),CV_32FC1) ;
        cvConvert(pSrc,variance) ;
        //cvSubS(variance,cv::Scalar(iris_mean),variance,safe_area) ;
		cvSubS(variance, iris_mean, variance, safe_area);
        cvMul(variance,variance,variance) ;
        //double iris_variance = sqrt(cvMean(variance,safe_area)) ;
		cv::Scalar irisvariance = cvAvg(variance, safe_area);
		double iris_variance = sqrt(irisvariance.val[0]);
        
        

        // Build mask of noise : |I-mean| > 2.35 * variance
        cv::Mat mask_noise = pSrc.clone() ;
        //cvAbsDiffS(pSrc,mask_noise,cv::Scalar(iris_mean)) ;
		cvAbsDiffS(pSrc, mask_noise, iris_mean);
        cvThreshold(mask_noise,mask_noise,2.35*iris_variance,255,CV_THRESH_BINARY) ;
        cv::bitwise_and(mask_iris,mask_noise,mask_noise) ;

        // Fusion with accurate contours
        cv::Mat accurate_contours = mask_iris.clone() ;
        struct_element = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_ELLIPSE) ;
        cvMorphologyEx(accurate_contours,accurate_contours,accurate_contours,struct_element,CV_MOP_GRADIENT) ;
        cvReleaseStructuringElement(&struct_element) ;
        reconstructMarkerByMask(accurate_contours,mask_noise,mask_noise) ;
        
        cvXor(mask_iris,mask_noise,pMask) ;
        
        // Release memory
        
        
        

    } // end of function




    void OsiProcessings::normalize ( const cv::Mat pSrc ,
                                           cv::Mat pDst ,
                                     const OsiCircle & rPupil ,
                                     const OsiCircle & rIris )
    {
        // Local variables
        cv::Point point_pupil , point_iris ;
        int x , y ;
        float theta , radius ;

        // Set to zeros all pixels
        pDst=0 ;

        // Loop on columns of normalized src
        for ( int j = 0 ; j < pDst->width ; j++ )
        {
            // One column correspond to an angle teta
            theta = (float) j / pDst->width * 2 * OSI_PI ;

            // Coordinates relative to both centers : iris and pupil
            point_pupil = convertPolarToCartesian(rPupil.getCenter(),rPupil.getRadius(),theta) ;
            point_iris = convertPolarToCartesian(rIris.getCenter(),rIris.getRadius(),theta) ;

            // Loop on lines of normalized src
            for ( int i = 0 ; i < pDst->height ; i++ )
            {    
                // The radial parameter
                radius = (float) i / pDst->height ;

                // Coordinates relative to both radii : iris and pupil
                x = (1-radius) * point_pupil.x + radius * point_iris.x ;
                y = (1-radius) * point_pupil.y + radius * point_iris.y ;

                // Do not exceed src size
                if ( x>=0 && x<pSrc->width && y>=0 && y<pSrc->height )
                {
                    ((uchar*)(pDst->imageData+i*pDst->widthStep))[j] = ((uchar*)(pSrc->imageData+y*pSrc->widthStep))[x] ;
                }
                
            }
        }    
    }

	// TODO : changer cette fonction pour normalisation avec contours
    void OsiProcessings::normalizeFromContour ( const cv::Mat pSrc ,
												      cv::Mat pDst ,
											    const OsiCircle & rPupil ,
												const OsiCircle & rIris ,
												const vector<float> rThetaCoarsePupil ,
												const vector<float> rThetaCoarseIris ,
												const vector<cv::Point> & rPupilCoarseContour ,
												const vector<cv::Point> & rIrisCoarseContour )
    {
        // Local variables
        cv::Point point_pupil , point_iris ;
        int x , y ;
        float theta , radius ;

        // Set to zeros all pixels
        pDst=0 ;

        // Loop on columns of normalized src
        for ( int j = 0 ; j < pDst->width ; j++ )
        {
            // One column correspond to an angle teta
            theta = (float) j / pDst->width * 2 * OSI_PI ;

			// Interpolate pupil and iris radii from coarse contours
			point_pupil = interpolate(rPupilCoarseContour,rThetaCoarsePupil,theta) ;
			point_iris = interpolate(rIrisCoarseContour,rThetaCoarseIris,theta) ;

            // Loop on lines of normalized src
            for ( int i = 0 ; i < pDst->height ; i++ )
            {    
                // The radial parameter
                radius = (float) i / pDst->height ;

                // Coordinates relative to both radii : iris and pupil
                x = (1-radius) * point_pupil.x + radius * point_iris.x ;
                y = (1-radius) * point_pupil.y + radius * point_iris.y ;

                // Do not exceed src size
                if ( x>=0 && x<pSrc->width && y>=0 && y<pSrc->height )
                {
                    ((uchar*)(pDst->imageData+i*pDst->widthStep))[j] = ((uchar*)(pSrc->imageData+y*pSrc->widthStep))[x] ;
                }
                
            }
        }    
    }


    cv::Point OsiProcessings::interpolate ( const vector<cv::Point> coarseContour ,
									      const vector<float> coarseTheta ,
									      const float theta )
    {
		float interpolation ;
		int i1 , i2 ;

		if ( theta < coarseTheta[0] )
		{
			i1 = coarseTheta.size() - 1 ;
			i2 = 0 ;
			interpolation = ( theta - (coarseTheta[i1]-2*OSI_PI) ) / ( coarseTheta[i2] - (coarseTheta[i1]-2*OSI_PI) ) ;
		}
			
		else if ( theta >= coarseTheta[coarseTheta.size()-1] )
		{
			i1 = coarseTheta.size() - 1 ;
			i2 = 0 ;
			interpolation = ( theta - coarseTheta[i1] ) / ( coarseTheta[i2]+2*OSI_PI - coarseTheta[i1] ) ;
		}

		else
		{
			int i = 0 ;
			while ( coarseTheta[i+1] <= theta ) i++ ;
			i1 = i ;
			i2 = i+1 ;
			interpolation = ( theta - coarseTheta[i1] ) / ( coarseTheta[i2] - coarseTheta[i1] ) ;			
		}
		

		float x = (1-interpolation) * coarseContour[i1].x + interpolation * coarseContour[i2].x ;
		float y = (1-interpolation) * coarseContour[i1].y + interpolation * coarseContour[i2].y ;
		
		return cv::Point(x,y) ;
	}


    void OsiProcessings::encode ( const cv::Mat pSrc ,
                                        cv::Mat pDst ,
                                  const vector<cv::Mat*> & rFilters )
    {
        // Compute the maximum width of the filters        
        int max_width = 0 ;
        for ( int f = 0 ; f < rFilters.size() ; f++ )
            if (rFilters[f]->cols > max_width)
                max_width = rFilters[f]->cols ;
        max_width = (max_width-1)/2 ;
        
        // Add wrapping borders on the left and right of image for convolution
        cv::Mat resized = addBorders(pSrc,max_width) ;

        // Temporary images to store the result of convolution
        cv::Mat img1 = cv::Mat(resized.size(),CV_32FC1) ;
        cv::Mat img2 = cv::Mat(resized.size(),pDst->depth,1) ;
        
        // Loop on filters
        for ( int f = 0 ; f < rFilters.size() ; f++ )
        {
            // Convolution
            cvFilter2D(resized,img1,rFilters[f]) ;

            // Threshold : above or below 0
            cvThreshold(img1,img2,0,255,CV_THRESH_BINARY) ;

            // Form the iris code
            cvSetImageROI(img2,cvRect(max_width,0,pSrc->width,pSrc->height)) ;
            cvSetImageROI(pDst,cvRect(0,f*pSrc->height,pSrc->width,pSrc->height)) ;
            cvCopy(img2,pDst,NULL) ;
            cvResetImageROI(img2) ;
            cvResetImageROI(pDst) ;
        }

        // Free memory
        
        
        
    }



    float OsiProcessings::match ( const cv::Mat image1 ,
                                  const cv::Mat image2 ,
                                  const cv::Mat mask )
    {    
        // Temporary matrix to store the XOR result
        cv::Mat result = cv::Mat(image1.size(),CV_8UC1) ;
        result=cv::Scalar(0) ;
        
        // Add borders on the image1 in order to shift it
        int shift = 10 ;
        cv::Mat shifted = addBorders(image1,shift) ;

        // The minimum score will be returned
        float score = 1 ;

        // Shift image1, and compare to image2
        for ( int s = -shift ; s <= shift ; s++ )
        {
            cvSetImageROI(shifted,cvRect(shift+s,0,image1->width,image1->height)) ;            
            cvXor(shifted,image2,result,mask) ;
            cvResetImageROI(shifted) ;
            float mean = (cvSum(result).val[0])/(cvSum(mask).val[0]) ;
            score = min(score,mean) ;
        }

        // Free memory
        
        

        return score ;
    }







    ///////////////////////////////////
    // PRIVATE METHODS
    ///////////////////////////////////


    // Convert polar coordinates into cartesian coordinates
    cv::Point OsiProcessings::convertPolarToCartesian ( const cv::Point & rCenter ,
                                                            int rRadius ,
                                                            float rTheta )
    {
        int x = rCenter.x + rRadius * cos(rTheta) ;
        int y = rCenter.y - rRadius * sin(rTheta) ;
        return cv::Point(x,y) ;
    }


    // Add left and right borders on an unwrapped image
    cv::Mat OsiProcessings::addBorders ( const cv::Mat pImage ,
                                                  int width )
    {
        // Result image
        cv::Mat result = cv::Mat(cv::Size(pImage->width+2*width,pImage->height),pImage->depth,pImage->nChannels) ;
        
        // Copy the image in the center
        cvCopyMakeBorder(pImage,result,cv::Point(width,0),IPL_BORDER_REPLICATE,cv::ScalarAll(0)) ;    

        // Create the borders left and right assuming wrapping
        for ( int i = 0 ; i < pImage->height ; i++ )
        {
            for ( int j = 0 ; j < width ; j++ )
            {
                ((uchar *)(result->imageData + i*result->widthStep))[j] = 
                ((uchar *)(pImage->imageData + i*pImage->widthStep))[pImage->width-width+j] ;
                ((uchar *)(result->imageData + i*result->widthStep))[result->width-width+j] = 
                ((uchar *)(pImage->imageData + i*pImage->widthStep))[j] ;
            }
        }

        return result ;
    }


    // Detect and locate a pupil inside an eye image
    void OsiProcessings::detectPupil ( const cv::Mat pSrc ,
                                             OsiCircle & rPupil ,
                                             int minPupilDiameter ,
                                             int maxPupilDiameter )
    {        
        // Check arguments
        //////////////////

        // String functions
        OsiStringUtils str ;

        // Default value for maxPupilDiameter, if user did not specify it
        if ( maxPupilDiameter == 0 )
        {
            maxPupilDiameter = min(pSrc->height,pSrc->width) * OSI_MAX_RATIO_PUPIL_IRIS ;
        }

        // Change maxPupilDiameter if it is too big relative to the image size and the ratio pupil/iris
        else if ( maxPupilDiameter > min(pSrc->height,pSrc->width) * OSI_MAX_RATIO_PUPIL_IRIS )
        {
            int newmaxPupilDiameter = floor(min(pSrc->height,pSrc->width)*OSI_MAX_RATIO_PUPIL_IRIS) ;
            cout << "Warning in function detectPupil : maxPupilDiameter = " << maxPupilDiameter ;
            cout << " is replaced by " << newmaxPupilDiameter ;
            cout << " because image size is " << pSrc->width << "x" << pSrc->height ;
            cout << " and ratio pupil/iris is generally lower than " << OSI_MAX_RATIO_PUPIL_IRIS << endl ;
            maxPupilDiameter = newmaxPupilDiameter ;
        }   
        
        // Change minPupilDiameter if it is too small relative to OSI_SMALLEST_PUPIL
        if ( minPupilDiameter < OSI_SMALLEST_PUPIL )
        {
            cout << "Warning in function detectPupil : minPupilDiameter = " << minPupilDiameter ;
            cout << " is replaced by " << OSI_SMALLEST_PUPIL ;
            cout << " which is the smallest size for detecting pupil" << endl ;
            minPupilDiameter = OSI_SMALLEST_PUPIL ;
        }

        // Check that minPupilDiameter < maxPupilDiameter
        if ( minPupilDiameter >= maxPupilDiameter )
        {
            throw invalid_argument("Error in function detectPupil : minPupilDiameter = " +
                                   str.toString(minPupilDiameter) +
                                   " should be lower than maxPupilDiameter = " +
                                   str.toString(maxPupilDiameter)) ;
        }       


        // Start processing
        ///////////////////

        // Resize image (downsample)
        float scale = (float) OSI_SMALLEST_PUPIL / minPupilDiameter ;
        cv::Mat resized = cv::Mat(cv::Size(pSrc->width*scale,pSrc->height*scale),pSrc->depth,1) ;
        cvResize(pSrc,resized) ;

        // Rescale sizes
        maxPupilDiameter = maxPupilDiameter * scale ;
        minPupilDiameter = minPupilDiameter * scale ;

        // Make sizes odd
        maxPupilDiameter += ( maxPupilDiameter % 2 ) ? 0 : +1 ;
        minPupilDiameter += ( minPupilDiameter % 2 ) ? 0 : -1 ;

        // Fill holes
        cv::Mat filled = cv::Mat(resized.size(),resized->depth,1) ;
        fillWhiteHoles(resized,filled) ;

        // Gradients in horizontal direction
        cv::Mat gh = cv::Mat(filled.size(),CV_32FC1) ;
        cvSobel(filled,gh,1,0) ;

        // Gradients in vertical direction
        cv::Mat gv = cv::Mat(filled.size(),CV_32FC1) ;
        cvSobel(filled,gv,0,1) ;

        // Normalize gradients
        cv::Mat gh2 = cv::Mat(filled.size(),CV_32FC1) ;
        cvMul(gh,gh,gh2) ;
        cv::Mat gv2 = cv::Mat(filled.size(),CV_32FC1) ;
        cvMul(gv,gv,gv2) ;
        cv::Mat gn = cv::Mat(filled.size(),CV_32FC1) ;        
        cvAdd(gh2,gv2,gn) ;
        cvPow(gn,gn,0.5) ;
        cvDiv(gh,gn,gh) ;
        cvDiv(gv,gn,gv) ;

        // Create the filters fh and fv
        int filter_size = maxPupilDiameter ;
        filter_size += ( filter_size % 2 ) ? 0 : -1 ;
        cv::Mat * fh = cv::Mat(filter_size,filter_size,CV_32FC1) ;
        cv::Mat * fv = cv::Mat(filter_size,filter_size,CV_32FC1) ;
        for ( int i = 0 ; i < fh->rows ; i++ )
        {
            float x = i - (filter_size-1)/2 ;
            for ( int j = 0 ; j < fh->cols ; j++ )
            {
                float y = j - (filter_size-1)/2 ;
                if ( x != 0 || y != 0 )
                {
                    (fh->data.fl)[i*fh->cols+j] = y / sqrt(x*x+y*y) ;
                    (fv->data.fl)[i*fv->cols+j] = x / sqrt(x*x+y*y) ;
                }
                else
                {
                    (fh->data.fl)[i*fh->cols+j] = 0 ;
                    (fv->data.fl)[i*fv->cols+j] = 0 ;
                }
            }
        }

        // Create the mask
        cv::Mat * mask = cv::Mat(filter_size,filter_size,CV_8UC1) ;

        // Temporary matrix for masking the filter (later : tempfilter = filter * mask)
        cv::Mat * temp_filter = cv::Mat(filter_size,filter_size,CV_32FC1) ;

        double old_max_val = 0 ;

        // Multi resolution of radius
        for ( int r = (OSI_SMALLEST_PUPIL-1)/2 ; r < (maxPupilDiameter-1)/2 ; r++ )
        {
            // Centred ring with radius = r and width = 2
            mask=0 ;
            cv::circle(*mask,cv::Point((filter_size-1)/2,(filter_size-1)/2),r,cv::Scalar(1),2) ;

            // Fh * Gh
            temp_filter=0 ;
            cvCopy(fh,temp_filter,mask) ;
            cvFilter2D(gh,gh2,temp_filter) ;

            // Fv * Gv
            temp_filter=0 ;
            cvCopy(fv,temp_filter,mask) ;
            cvFilter2D(gv,gv2,temp_filter) ;

            // Fh*Gh + Fv*Gv
            cvAdd(gh2,gv2,gn) ;
            cvScale(gn,gn,1.0/cvSum(mask).val[0]) ;

            // Sum in the disk-shaped neighbourhood
            mask=0 ;
            cv::circle(mask,cv::Point((filter_size-1)/2,(filter_size-1)/2),r,cv::Scalar(1),-1) ;
            cvFilter2D(filled,gh2,mask) ;
            cvScale(gh2,gh2,-1.0/cvSum(mask).val[0]/255.0,1) ;

            // Add the two features : contour + darkness
            cvAdd(gn,gh2,gn) ;

            // Find the maximum in feature image
            double max_val ;
            cv::Point max_loc ;
            cvMinMaxLoc(gn,0,&max_val,0,&max_loc) ;

            if ( max_val > old_max_val )
            {
                old_max_val = max_val ;
                rPupil.setCircle(max_loc,r) ;
            }
        }

        // Rescale circle        
        int x = ( (float) ( rPupil.getCenter().x * (pSrc->width-1) ) ) / (filled->width-1) + (float)((1.0/scale)-1)/2  ;
        int y = ( (float) ( rPupil.getCenter().y * (pSrc->height-1) ) ) / (filled->height-1) + (float)((1.0/scale)-1)/2 ;
        int r = rPupil.getRadius() / scale ;
        rPupil.setCircle(x,y,r) ;
        
        // Release memory 

    } // end of function




    // Morphological reconstruction
    void OsiProcessings::reconstructMarkerByMask ( const cv::Mat pMarker ,
                                                   const cv::Mat pMask ,
                                                         cv::Mat pDst )
    {
        // Temporary image that will inform about marker evolution
        cv::Mat difference = pMask.clone() ;

        // :WARNING: if user calls f(x,y,y) instead of f(x,y,z), the mask MUST be cloned before processing
        cv::Mat mask = pMask.clone() ;

        // Copy the marker
        cvCopy(pMarker,pDst) ;

        // Structuring element for morphological operation
        IplConvKernel * structuring_element = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_ELLIPSE) ;

        // Will stop when marker does not change anymore
        while ( cvSum(difference).val[0] )
        {
            // Remind marker before processing, in order to
            // compare with the marker after processing
            cvCopy(pDst,difference) ;

            // Dilate the marker
            cvDilate(pDst,pDst,structuring_element) ;

            // Keep the minimum between marker and mask
            cvMin(pDst,mask,pDst) ;

            // Evolution of the marker
            cvAbsDiff(pDst,difference,difference) ;          
        }

        // Release memory
        
        
        cvReleaseStructuringElement(&structuring_element) ;

    } // end of function




    // Fill the white holes surrounded by dark pixels, such as specular reflection inside pupil area
    void OsiProcessings::fillWhiteHoles ( const cv::Mat pSrc ,
                                                cv::Mat pDst )
    {
        int width , height ;
        if ( pSrc->roi )
        {
            width = pSrc->roi->width ;
            height = pSrc->roi->height ;
        }
        else
        {
            width = pSrc->width ;
            height = pSrc->height ;
        }

        // Mask for reconstruction : pSrc + borders=0
        cv::Mat mask = cv::Mat(cv::Size(width+2,height+2),pSrc->depth,1) ;
        mask=0 ;
        cvSetImageROI(mask,cvRect(1,1,width,height)) ;
        cvCopy(pSrc,mask) ;
        cvResetImageROI(mask) ;

        // Marker for reconstruction : all=0 + borders=255
        cv::Mat marker = mask.clone() ;
        marker=0 ;
        cvRectangle(marker,cv::Point(1,1),cv::Point(width+1,height+1),cv::Scalar(255)) ;

        // Temporary result of reconstruction
        cv::Mat result = mask.clone() ;

        // Morphological reconstruction
        reconstructMarkerByMask(marker,mask,result) ;

        // Remove borders
        cvSetImageROI(result,cvRect(1,1,width,height)) ;
        cvCopy(result,pDst) ;

        // Release memory
        
        
        

    } // end of function



    // Rescale between 0 and 255, and show image
    void OsiProcessings::showImage ( const cv::Mat pImage ,
                                           int delay ,
                                     const string & rWindowName )
    {
        cv::Mat show ;

        if ( pImage->nChannels == 1 )
        {
            // Rescale between 0 and 255 by computing : (X-min)/(max-min)
            double min_val , max_val ;
            cvMinMaxLoc(pImage,&min_val,&max_val) ;
            cv::Mat scaled = pImage.clone() ;
            cvScale(pImage,scaled,255/(max_val-min_val),-min_val/(max_val-min_val)) ;

            // Convert into 8-bit
            show = cv::Mat(pImage.size(),CV_8UC1) ;
            cvConvert(scaled,show) ;

            // Release memory
            
        }
        else
        {
            show = pImage.clone() ;
        }            

        // Show image
        cvShowImage(rWindowName.c_str(),show) ;
        cvWaitKey(delay) ;

        // Release image        
        
    }





    // Unwrap a ring into a rectangular band
    cv::Mat OsiProcessings::unwrapRing ( const cv::Mat pSrc ,
                                            const cv::Point & rCenter ,
                                                  int minRadius ,
                                                  int maxRadius ,
                                            const vector<float> & rTheta )
    {
        // Result image
        cv::Mat result = cv::Mat(cv::Size(rTheta.size(),maxRadius-minRadius+1),pSrc->depth,1) ;
        result=0 ;

        // Loop on columns of normalized image
        for ( int j = 0 ; j < result->width ; j++ )
        {
            // Loop on lines of normalized image
            for ( int i = 0 ; i < result->height ; i++ )
            {
                cv::Point point = convertPolarToCartesian(rCenter,minRadius+i,rTheta[j]) ;

                // Do not exceed image size
                if ( point.x >= 0 && point.x < pSrc->width && point.y >= 0 && point.y < pSrc->height )
                    ((uchar *)(result->imageData+i*result->widthStep))[j] =
                    ((uchar *)(pSrc->imageData+point.y*pSrc->widthStep))[point.x] ;
            }
        }
        return result ;
    } 







    // Smooth the image by anisotropic smoothing (Gross & Brajovic,2003)
    void OsiProcessings::processAnisotropicSmoothing ( const cv::Mat pSrc ,
                                                             cv::Mat pDst ,
                                                             int iterations ,
                                                             float lambda )
    {
        // Temporary float images
        cv::Mat tfs = cv::Mat(pSrc.size(),CV_32FC1) ;
        cvConvert(pSrc,tfs) ;
        cv::Mat tfd = cv::Mat(pSrc.size(),CV_32FC1) ;        
        cvConvert(pSrc,tfd) ;

        // Make borders dark
        cvRectangle(tfd,cv::Point(0,0),cv::Point(tfd->width-1,tfd->height-1),cv::Scalar(0)) ;

        // Weber coefficients
        float rhon , rhos , rhoe , rhow ;

        // Store pixel values
        float tfsc , tfsn , tfss , tfse , tfsw , tfdn , tfds , tfde , tfdw ;

        // Loop on iterations
        for ( int k = 0 ; k < iterations ; k++ )
        {
            // Odd pixels
            for ( int i = 1 ; i < tfs->height-1 ; i++ )
            {
                for ( int j = 2-i%2 ; j < tfs->width-1 ; j = j + 2 )
                {
                    // Get pixels in neighbourhood of original image
                    tfsc = ((float*)(tfs->imageData+i*tfs->widthStep))[j] ;
                    tfsn = ((float*)(tfs->imageData+(i-1)*tfs->widthStep))[j] ;
                    tfss = ((float*)(tfs->imageData+(i+1)*tfs->widthStep))[j] ;
                    tfse = ((float*)(tfs->imageData+i*tfs->widthStep))[j-1] ;
                    tfsw = ((float*)(tfs->imageData+i*tfs->widthStep))[j+1] ;                
                    
                    // Get pixels in neighbourhood of light image
                    tfdn = ((float*)(tfd->imageData+(i-1)*tfd->widthStep))[j] ;
                    tfds = ((float*)(tfd->imageData+(i+1)*tfd->widthStep))[j] ;
                    tfde = ((float*)(tfd->imageData+i*tfd->widthStep))[j-1] ;
                    tfdw = ((float*)(tfd->imageData+i*tfd->widthStep))[j+1] ;                    

                    // Compute weber coefficients
                    rhon = min(tfsn,tfsc) / max<float>(1.0,abs(tfsn-tfsc)) ;
                    rhos = min(tfss,tfsc) / max<float>(1.0,abs(tfss-tfsc)) ;
                    rhoe = min(tfse,tfsc) / max<float>(1.0,abs(tfse-tfsc)) ;
                    rhow = min(tfsw,tfsc) / max<float>(1.0,abs(tfsw-tfsc)) ;                    

                    // Compute LightImage(i,j)                    
                    ((float*)(tfd->imageData+i*tfd->widthStep))[j] = ( ( tfsc + lambda *
                    ( rhon * tfdn + rhos * tfds + rhoe * tfde + rhow * tfdw ) )
                    / ( 1 + lambda * ( rhon + rhos + rhoe + rhow ) ) ) ;
                }
            }

            cvCopy(tfd,tfs) ;

            // Even pixels
            for ( int i = 1 ; i < tfs->height-1 ; i++ )
            {
                for ( int j = 1+i%2 ; j < tfs->width-1 ; j = j + 2 )
                {
                    // Get pixels in neighbourhood of original image
                    tfsc = ((float*)(tfs->imageData+i*tfs->widthStep))[j] ;
                    tfsn = ((float*)(tfs->imageData+(i-1)*tfs->widthStep))[j] ;
                    tfss = ((float*)(tfs->imageData+(i+1)*tfs->widthStep))[j] ;
                    tfse = ((float*)(tfs->imageData+i*tfs->widthStep))[j-1] ;
                    tfsw = ((float*)(tfs->imageData+i*tfs->widthStep))[j+1] ;                
                    
                    // Get pixels in neighbourhood of light image
                    tfdn = ((float*)(tfd->imageData+(i-1)*tfd->widthStep))[j] ;
                    tfds = ((float*)(tfd->imageData+(i+1)*tfd->widthStep))[j] ;
                    tfde = ((float*)(tfd->imageData+i*tfd->widthStep))[j-1] ;
                    tfdw = ((float*)(tfd->imageData+i*tfd->widthStep))[j+1] ;                    

                    // Compute weber coefficients
                    rhon = min(tfsn,tfsc) / max<float>(1.0,abs(tfsn-tfsc)) ;
                    rhos = min(tfss,tfsc) / max<float>(1.0,abs(tfss-tfsc)) ;
                    rhoe = min(tfse,tfsc) / max<float>(1.0,abs(tfse-tfsc)) ;
                    rhow = min(tfsw,tfsc) / max<float>(1.0,abs(tfsw-tfsc)) ;                    

                    // Compute LightImage(i,j)                    
                    ((float*)(tfd->imageData+i*tfd->widthStep))[j] = ( ( tfsc + lambda *
                    ( rhon * tfdn + rhos * tfds + rhoe * tfde + rhow * tfdw ) )
                    / ( 1 + lambda * ( rhon + rhos + rhoe + rhow ) ) ) ;
                }
            }

            cvCopy(tfd,tfs) ;
            cvConvert(tfd,pDst) ;

        } // end of iterations k

        // Borders of image
        for ( int i = 0 ; i < tfd->height ; i++ )
        {
            ((uchar*)(pDst->imageData+i*pDst->widthStep))[0] =
            ((uchar*)(pDst->imageData+i*pDst->widthStep))[1] ;
            ((uchar*)(pDst->imageData+i*pDst->widthStep))[pDst->width-1] =
            ((uchar*)(pDst->imageData+i*pDst->widthStep))[pDst->width-2] ;
        }
        for ( int j = 0 ; j < tfd->width ; j++ )
        {
            ((uchar*)(pDst->imageData))[j] =
            ((uchar*)(pDst->imageData+pDst->widthStep))[j] ;
            ((uchar*)(pDst->imageData+(pDst->height-1)*pDst->widthStep))[j] =
            ((uchar*)(pDst->imageData+(pDst->height-2)*pDst->widthStep))[j] ;
        }

        // Release memory
        
        

    } // end of function






    // Compute vertical gradients using Sobel operator
    void OsiProcessings::computeVerticalGradients ( const cv::Mat pSrc , cv::Mat pDst )
    {
        // Float values for Sobel
        cv::Mat result_sobel = cv::Mat(pSrc.size(),CV_32FC1) ;
        
        // Sobel filter in vertical direction
        cvSobel(pSrc,result_sobel,0,1) ;

        // Remove negative edges, ie from white (top) to black (bottom)
        cvThreshold(result_sobel,result_sobel,0,0,CV_THRESH_TOZERO) ;

        // Convert into 8-bit
        double min , max ;
        cvMinMaxLoc(result_sobel,&min,&max) ;
        cvConvertScale(result_sobel,pDst,255/(max-min),-255*min/(max-min)) ;

        // Release memory
        

    } // end of function






    // Run viterbi algorithm on gradient (or probability) image and find optimal path
    void OsiProcessings::runViterbi ( const cv::Mat pSrc , vector<int> & rOptimalPath )
    {
        // Initialize the output
        rOptimalPath.clear() ;
        rOptimalPath.resize(pSrc->width) ;
        
        // Initialize cost matrix to zero
        cv::Mat cost = cv::Mat(pSrc.size(),CV_32FC1) ;
        cost=0 ;

        // Forward process : build the cost matrix
        for ( int w = 0 ; w < pSrc->width ; w++ )
        {
            for ( int h = 0 ; h < pSrc->height ; h++ )
            {
                // First column is same as source image
                if ( w == 0 )
                    ((float*)(cost->imageData+h*cost->widthStep))[w] =
                    ((uchar*)(pSrc->imageData+h*pSrc->widthStep))[w] ;

                else
                {
                    // First line
                    if ( h == 0 )
                        ((float*)(cost->imageData+h*cost->widthStep))[w] = max<float>(
                        ((float*)(cost->imageData+(h)*cost->widthStep))[w-1],
                        ((float*)(cost->imageData+(h+1)*cost->widthStep))[w-1]) +
                        ((uchar*)(pSrc->imageData+h*pSrc->widthStep))[w] ;

                    // Last line
                    else if ( h == pSrc->height - 1 )
                    {
                        ((float*)(cost->imageData+h*cost->widthStep))[w] = max<float>(
                        ((float*)(cost->imageData+h*cost->widthStep))[w-1],
                        ((float*)(cost->imageData+(h-1)*cost->widthStep))[w-1]) +
                        ((uchar*)(pSrc->imageData+h*pSrc->widthStep))[w] ;
                    }

                    // Middle lines
                    else
                        ((float*)(cost->imageData+h*cost->widthStep))[w] = max<float>(
                        ((float*)(cost->imageData+h*cost->widthStep))[w-1],max<float>(
                        ((float*)(cost->imageData+(h+1)*cost->widthStep))[w-1],
                        ((float*)(cost->imageData+(h-1)*cost->widthStep))[w-1])) +
                        ((uchar*)(pSrc->imageData+h*pSrc->widthStep))[w] ;
                }
            }
        }

        // Get the maximum in last column of cost matrix
        cvSetImageROI(cost,cvRect(cost->width-1,0,1,cost->height)) ;
        cv::Point max_loc ;        
        cvMinMaxLoc(cost,0,0,0,&max_loc) ;
        int h = max_loc.y ;
        int h0 = h ;
        cvResetImageROI(cost) ;        

        // Store the point in output vector
        rOptimalPath[rOptimalPath.size()-1] = h0 ;

        float h1 , h2 , h3 ;

        // Backward process
        for ( int w = rOptimalPath.size() - 2 ; w >= 0 ; w-- )
        {
            // Constraint to close the contour
            if ( h - h0 > w )
                h -- ;
            else if ( h0 - h > w )
                h ++ ;

            // When no need to constraint : use the cost matrix
            else
            {
                // h1 is the value above line h
                h1 = ( h == 0 ) ? 0 : ((float*)(cost->imageData+(h-1)*cost->widthStep))[w] ;

                // h2 is the value at line h
                h2 = ((float*)(cost->imageData+h*cost->widthStep))[w] ;

                // h3 is the value below line h
                h3 = ( h == cost->height - 1 ) ? 0 : ((float*)(cost->imageData+(h+1)*cost->widthStep))[w] ;
                
                // h1 is the maximum => h decreases
                if ( h1 > h2 && h1 > h3 )
                    h-- ;

                // h3 is the maximum => h increases
                else if ( h3 > h2 && h3 > h1 )
                    h++ ;
            }

            // Store the point in output contour
            rOptimalPath[w] = h ;

        }

        // Release memory
        

    } // end of function



    // Find a contour in image using Viterbi algorithm and anisotropic smoothing
    vector<cv::Point> OsiProcessings::findContour ( const cv::Mat pSrc ,
                                                  const cv::Point & rCenter ,
                                                  const vector<float> & rTheta ,
                                                        int minRadius ,
                                                        int maxRadius ,
                                                  const cv::Mat pMask )
    {
        // Output
        vector<cv::Point> contour ;
        contour.resize(rTheta.size()) ;

        // Unwrap the image
        cv::Mat unwrapped = unwrapRing(pSrc,rCenter,minRadius,maxRadius,rTheta) ;

        // Smooth image
        processAnisotropicSmoothing(unwrapped,unwrapped,100,1) ;

        // Extract the gradients
        computeVerticalGradients(unwrapped,unwrapped) ;

        // Take into account the mask
        if ( pMask )
        {
            cv::Mat mask_unwrapped = unwrapRing(pMask,rCenter,minRadius,maxRadius,rTheta) ;
            cv::Mat temp = unwrapped.clone() ;
            unwrapped=0 ;
            cvCopy(temp,unwrapped,mask_unwrapped) ;
            
            
        }

        // Find optimal path in unwrapped image
        vector<int> optimalPath ;
        runViterbi(unwrapped,optimalPath) ;
        for ( int i = 0 ; i < optimalPath.size() ; i++ )
        {
            contour[i] = convertPolarToCartesian(rCenter,minRadius+optimalPath[i],rTheta[i]) ;
        }

        // Release memory
        

        return contour ;

    } // end of function



    // Draw a contour (vector of cv::Point) on an image
    void OsiProcessings::drawContour ( cv::Mat pImage , const vector<cv::Point> & rContour , const cv::Scalar & rColor , int thickness )
    {
        // Draw INSIDE the contour if thickness is negative
        if ( thickness < 0 )
        {
            cv::Point * points = new cv::Point[rContour.size()] ;
            for ( int i = 0 ; i < rContour.size() ; i++ )
            {
                points[i].x = rContour[i].x ;
                points[i].y = rContour[i].y ;
            }
            cvFillConvexPoly(pImage,points,rContour.size(),rColor) ;
            delete [] points ;
        }

        // Else draw the contour
        else
        {
            // Draw the contour on binary mask
            cv::Mat mask = cv::Mat(pImage.size(),CV_8UC1) ;
            mask=0 ;
            for ( int i = 0 ; i < rContour.size() ; i++ )
            {
                // Do not exceed image sizes
                int x = min(max(0,rContour[i].x),pImage->width) ;
                int y = min(max(0,rContour[i].y),pImage->height) ;

                // Plot the point on image
                ((uchar *)(mask->imageData+y*mask->widthStep))[x] = 255 ;
            }
        
            // Dilate mask if user specified thickness
            if ( thickness > 1 )
            {
                IplConvKernel * se = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_ELLIPSE) ;
                cvDilate(mask,mask,se,thickness-1) ;
                cvReleaseStructuringElement(&se) ;
            }

            // Color rgb
            cvSet(pImage,rColor,mask) ;

            // Release memory
            

        }

    } // end of function



} // end of namespace


