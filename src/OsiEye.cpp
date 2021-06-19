/*******************************************************
* Open Source for Iris : OSIRIS
* Version : 4.0
* Date : 2011
* Author : Guillaume Sutra, Telecom SudParis, France
* License : BSD
********************************************************/

#include <fstream>
#include <stdexcept>
#include "opencv2/opencv.hpp"
#include "OsiEye.h"
#include "OsiProcessings.h"

using namespace std ;

namespace osiris
{

    // CONSTRUCTORS & DESTRUCTORS
    /////////////////////////////

    OsiEye::OsiEye ( )
    {
        mpOriginalImage = 0 ;        
        mpSegmentedImage = 0 ;
        mpMask = 0 ;
        mpNormalizedImage = 0 ;
        mpNormalizedMask = 0 ;
        mpIrisCode = 0 ;
        mPupil.setCircle(0,0,0) ;
        mIris.setCircle(0,0,0) ;
    }

    OsiEye::~OsiEye ( )
    {
        
        
        
        
        
        
    }





    // Functions for loading images and parameters
    //////////////////////////////////////////////

    void OsiEye::loadImage ( const string & rFilename , cv::Mat* ppImage )
    {
        // :WARNING: ppImage is a pointer of pointer
        try
        {
            if ( ppImage->empty() )
            {
                
            }

            *ppImage = cv::imread(rFilename.c_str(),0) ;
            if ( ppImage->empty() )
            {
                cout << "Cannot load image : " << rFilename << endl ;
            }
        }
        catch ( exception & e )
        {
            cout << e.what() << endl ;
        }
    }



    void OsiEye::loadOriginalImage ( const string & rFilename )
    {
        loadImage(rFilename,&mpOriginalImage) ;
    }



    void OsiEye::loadMask ( const string & rFilename )
    {
        loadImage(rFilename,&mpMask) ;
    }



    void OsiEye::loadNormalizedImage ( const string & rFilename )
    {
        loadImage(rFilename,&mpNormalizedImage) ;
    }



    void OsiEye::loadNormalizedMask ( const string & rFilename )
    {
        loadImage(rFilename,&mpNormalizedMask) ;
    }



    void OsiEye::loadIrisCode ( const string & rFilename )
    {
        loadImage(rFilename,&mpIrisCode) ;
    }



    void OsiEye::loadParameters (const string & rFilename )
    {
        // Open the file
        ifstream file(rFilename.c_str(),ios::in) ;

        // If file is not opened
        if ( ! file )
        {
            throw runtime_error("Cannot load the parameters in " + rFilename) ;
        }
        try
        {            
            //int xp , yp , rp , xi , yi , ri ;
            //file >> xp ;
            //file >> yp ;
            //file >> rp ;
            //file >> xi ;
            //file >> yi ;
            //file >> ri ;
            //mPupil.setCircle(xp,yp,rp) ;
            //mIris.setCircle(xi,yi,ri) ;
			int nbp = 0 ;
			int nbi = 0 ;
			file >> nbp ;
			file >> nbi  ;
			mThetaCoarsePupil.resize(nbp, 0.0) ;
			mThetaCoarseIris.resize(nbi, 0.0) ;
			mCoarsePupilContour.resize(nbp, cv::Point(0,0)) ;
			mCoarseIrisContour.resize(nbi ,cv::Point(0,0)) ;
			//matrix.resize( num_of col , vector<double>( num_of_row , init_value ) );
			for (int i = 0 ; i < nbp ; i++)
			{
				file >> mCoarsePupilContour[i].x ;
				file >> mCoarsePupilContour[i].y ;
				file >> mThetaCoarsePupil[i] ;
			}
			for (int j = 0 ; j < nbi ; j++)
			{
				file >> mCoarseIrisContour[j].x ;
				file >> mCoarseIrisContour[j].y ;
				file >> mThetaCoarseIris[j] ;
			}
			
        }
        catch ( exception & e )
        {
            cout << e.what() << endl ;
            throw runtime_error("Error while loading parameters from " + rFilename) ;
        }

        // Close the file
        file.close() ;
    }






    // Functions for saving images and parameters
    /////////////////////////////////////////////



    void OsiEye::saveImage ( const string & rFilename , const cv::Mat pImage )
    {
        // :TODO: no exception here, but 2 error messages
        // 1. pImage does NOT exist => "image was neither comptued nor loaded"
        // 2. cvSaveImage returns <=0 => "rFilename = invalid for saving"
        if (  pImage.empty() )
        {
            throw runtime_error("Cannot save image " + rFilename + " because this image is not built") ;
        }
        if ( ! cv::imwrite(rFilename.c_str(),pImage) )
        {
            cout << "Cannot save image as " << rFilename << endl ;
        }
    }



    void OsiEye::saveSegmentedImage ( const string & rFilename )
    {        
        saveImage(rFilename,mpSegmentedImage) ;
    }



    void OsiEye::saveMask ( const string & rFilename )
    {
        saveImage(rFilename,mpMask) ;
    }



    void OsiEye::saveNormalizedImage ( const string & rFilename )
    {
        saveImage(rFilename,mpNormalizedImage) ;
    }



    void OsiEye::saveNormalizedMask ( const string & rFilename )
    {
        saveImage(rFilename,mpNormalizedMask) ;
    }



    void OsiEye::saveIrisCode ( const string & rFilename )
    {
        saveImage(rFilename,mpIrisCode) ;
    }



    void OsiEye::saveParameters (const string & rFilename )
    {
        // Open the file
        ofstream file(rFilename.c_str(),ios::out) ;

        // If file is not opened
        if ( ! file )
        {
            throw runtime_error("Cannot save the parameters in " + rFilename) ;
        }
        
        try
        {
        //    file << mPupil.getCenter().x << " " ;
        //    file << mPupil.getCenter().y << " " ;
        //    file << mPupil.getRadius() << endl ;
        //    file << mIris.getCenter().x << " " ;
        //    file << mIris.getCenter().y << " " ;
        //    file << mIris.getRadius() << endl ;
			file << mCoarsePupilContour.size() << endl ;
			file << mCoarseIrisContour.size() << endl ;
			for (int i=0; i<(mCoarsePupilContour.size()); i++)
			{
				file << mCoarsePupilContour[i].x << " " ;
				file << mCoarsePupilContour[i].y << " " ;
				file << mThetaCoarsePupil[i] << " " ;
			}
			file << endl ;
			for (int j=0; j<(mCoarseIrisContour.size()); j++)
			{	
				file << mCoarseIrisContour[j].x << " " ;
				file << mCoarseIrisContour[j].y << " " ;
				file << mThetaCoarseIris[j] << " " ;
			}
        }
        catch ( exception & e )
        {
            cout << e.what() << endl ;
            throw runtime_error("Error while saving parameters in " + rFilename) ;
        }

        // Close the file
        file.close() ;
    }







    // Functions for processings
    ////////////////////////////



    void OsiEye::initMask ( )
    {
        if ( mpMask.empty() )
        {
            
        }
        if ( ! mpOriginalImage.empty() )
        {
            throw runtime_error("Cannot initialize the mask because original image is not loaded") ;
        }
        mpMask = cv::Mat(mpOriginalImage.size(),CV_8UC1) ;
        mpMask=cv::Scalar(255) ;
    }



    void OsiEye::segment ( int minIrisDiameter , int minPupilDiameter , int maxIrisDiameter , int maxPupilDiameter )
    {
        if ( ! mpOriginalImage.empty() )
        {
            throw runtime_error("Cannot segment image because original image is not loaded") ;
        }

        // Initialize mask and segmented image
        mpMask = cv::Mat(mpOriginalImage.size(),CV_8UC1) ;
        mpSegmentedImage = cv::Mat(mpOriginalImage.size(),CV_8UC3) ;
        cvCvtColor(mpOriginalImage,mpSegmentedImage,CV_GRAY2BGR) ;

        // Processing functions
        OsiProcessings op ;

        // Segment the eye
        op.segment(mpOriginalImage,mpMask,mPupil,mIris,mThetaCoarsePupil,mThetaCoarseIris,mCoarsePupilContour,mCoarseIrisContour,minIrisDiameter,minPupilDiameter,maxIrisDiameter,maxPupilDiameter) ;

        // Draw on segmented image
        cv::Mat tmp = cvCloneImage(mpMask) ;
        tmp=0 ;
        cv::circle(tmp,mIris.getCenter(),mIris.getRadius(),cv::Scalar(255),-1) ;
        cv::circle(tmp,mPupil.getCenter(),mPupil.getRadius(),cv::Scalar(0),-1) ;
        cvSub(tmp,mpMask,tmp) ;
        cvSet(mpSegmentedImage,cv::Scalar(0,0,255),tmp) ;
        
        cv::circle(mpSegmentedImage,mPupil.getCenter(),mPupil.getRadius(),cv::Scalar(0,255,0)) ;
        cv::circle(mpSegmentedImage,mIris.getCenter(),mIris.getRadius(),cv::Scalar(0,255,0)) ;

    }



    void OsiEye::normalize ( int rWidthOfNormalizedIris , int rHeightOfNormalizedIris )
    {
        // Processing functions
        OsiProcessings op ;

        // For the image
        if ( ! mpOriginalImage.empty() )
        {
            throw runtime_error("Cannot normalize image because original image is not loaded") ;
        }
    
        mpNormalizedImage = cv::Mat(cvSize(rWidthOfNormalizedIris,rHeightOfNormalizedIris),CV_8UC1) ;

        if ( mThetaCoarsePupil.empty() || mThetaCoarseIris.empty() )
        {
            //throw runtime_error("Cannot normalize image because circles are not correctly computed") ;
			throw runtime_error("Cannot normalize image because contours are not correctly computed/loaded") ;
        }
        
        //op.normalize(mpOriginalImage,mpNormalizedImage,mPupil,mIris) ;
		op.normalizeFromContour(mpOriginalImage,mpNormalizedImage,mPupil,mIris,mThetaCoarsePupil,mThetaCoarseIris,mCoarsePupilContour,mCoarseIrisContour) ;

        // For the mask
        if ( ! mpMask )
        {
            initMask() ;
        }

        mpNormalizedMask = cv::Mat(cvSize(rWidthOfNormalizedIris,rHeightOfNormalizedIris),CV_8UC1) ;
        
        //op.normalize(mpMask,mpNormalizedMask,mPupil,mIris) ;
		op.normalizeFromContour(mpMask,mpNormalizedMask,mPupil,mIris,mThetaCoarsePupil,mThetaCoarseIris,mCoarsePupilContour,mCoarseIrisContour) ;
    }



    void OsiEye::encode ( const vector<cv::Mat*> & rGaborFilters )
    {
        if ( ! mpNormalizedImage.empty() )
        {
            throw runtime_error("Cannot encode because normalized image is not loaded") ;
        }

        // Create the image to store the iris code
        CvSize size = mpNormalizedImage.size() ;
        mpIrisCode = cv::Mat(cvSize(size.width,size.height*rGaborFilters.size()),CV_8UC1) ;

        // Encode
        OsiProcessings op ;
        op.encode(mpNormalizedImage,mpIrisCode,rGaborFilters) ;
    }



    float OsiEye::match ( OsiEye & rEye , const cv::Mat * pApplicationPoints )
    {
        // Check that both iris codes are built
        if ( ! mpIrisCode.empty() )
        {
            throw runtime_error("Cannot match because iris code 1 is not built (nor computed neither loaded)") ;
        }
        if ( ! rEye.mpIrisCode.empty() )
        {
            throw runtime_error("Cannot match because iris code 2 is not built (nor computed neither loaded)") ;
        }

        // Initialize the normalized masks
        // :TODO: must inform the user of this step, for example if user provides masks for all images
        // but one is missing for only one image. However, message must not be spammed if the user
        // did not provide any mask ! So it must be found a way to inform user but without spamming
        if ( ! mpNormalizedMask.empty() )
        {
            mpNormalizedMask = cv::Mat(pApplicationPoints.size(),CV_8UC1) ;
            mpNormalizedMask=cv::Scalar(255) ;
            //cout << "Normalized mask of image 1 is missing for matching. All pixels are initialized to 255" << endl ;
        }
        if ( ! rEye.mpNormalizedMask.empty() )
        {
            rEye.mpNormalizedMask = cv::Mat(pApplicationPoints.size(),CV_8UC1) ;
            rEye.mpNormalizedMask=cv::Scalar(255) ;
            //cout << "Normalized mask of image 2 is missing for matching. All pixels are initialized to 255" << endl ;
        }

        // Build the total mask = mask1 * mask2 * points    
        cv::Mat temp = cv::Mat(pApplicationPoints.size(),mpIrisCode->depth,1) ;
        temp=cv::Scalar(0) ;
        cvAnd(mpNormalizedMask,rEye.mpNormalizedMask,temp,pApplicationPoints) ;

        // Copy the mask f times, where f correspond to the number of codes (= number of filters)
        int n_codes = mpIrisCode->height / pApplicationPoints->height ;
        cv::Mat total_mask = cv::Mat(mpIrisCode.size(),CV_8UC1) ;
        for ( int n = 0 ; n < n_codes ; n++ )
        {
            cvSetImageROI(total_mask,cvRect(0,n*pApplicationPoints->height,pApplicationPoints->width,pApplicationPoints->height)) ;
            cvCopy(temp,total_mask) ;        
            cvResetImageROI(total_mask) ;
        }

        // Match
        OsiProcessings op ;
        float score = op.match(mpIrisCode,rEye.mpIrisCode,total_mask) ;

        // Free memory
        
        
    
        return score ;
    }



} // end of namespace