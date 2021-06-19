/*******************************************************
* Open Source for Iris : OSIRIS
* Version : 4.0
* Date : 2011
* Author : Guillaume Sutra, Telecom SudParis, France
* License : BSD
********************************************************/

#include <stdexcept>
#include "OsiCircle.h"

using namespace std ;

namespace osiris
{

    // CONSTRUCTORS & DESTRUCTORS
    /////////////////////////////


    OsiCircle::OsiCircle()
    {
        // Do nothing
    }

    OsiCircle::~OsiCircle()
    {
        // Do nothing
    }

    OsiCircle::OsiCircle ( const cv::Point & rCenter , int rRadius )
    {
        setCenter(rCenter) ;
        setRadius(rRadius) ;
    }




    // ACCESSORS
    ////////////


    cv::Point OsiCircle::getCenter ( ) const
    {
        return mCenter ;
    }

    int OsiCircle::getRadius ( ) const
    {
        return mRadius ;
    }

    void OsiCircle::setCenter(const cv::Point & rCenter )
    {
        mCenter = rCenter ;
    }

    void OsiCircle::setRadius ( int rRadius )
    {
        if ( rRadius < 0 )
        {
            throw runtime_error("Circle with negative radius : " + rRadius) ;
        }
        mRadius = rRadius ;
    }

    void OsiCircle::setCircle ( const cv::Point & rCenter , int rRadius )
    {
        setCenter(rCenter) ;
        setRadius(rRadius) ;
    }

    void OsiCircle::setCircle ( int rCenterX , int rCenterY , int rRadius )
    {
        setCircle(cv::Point(rCenterX,rCenterY),rRadius) ;
    }





    // OPERATORS
    ////////////


    void OsiCircle::drawCircle ( cv::Mat pImage , const cv::Scalar & rColor , int thickness )
    {
        cv::circle(pImage,mCenter,mRadius,rColor,thickness) ;
    }


    void OsiCircle::computeCircleFitting ( const vector<cv::Point> & rPoints )
    {
        // Compute the averages mx and my
        float mx = 0 , my = 0 ;
        for ( int p = 0 ; p < rPoints.size() ; p++ )
        {
            mx += rPoints[p].x ;
            my += rPoints[p].y ;
        }
        mx = mx / rPoints.size() ;
        my = my / rPoints.size() ;

        // Work in (u,v) space, with u = x-mx and v = y-my
        float u = 0 , v = 0 , suu = 0 , svv = 0 , suv = 0 , suuu = 0 , svvv = 0 , suuv = 0 , suvv = 0 ;

        // Build some sums
        for ( int p = 0 ; p < rPoints.size() ; p++ )
        {
            u = rPoints[p].x - mx ;
            v = rPoints[p].y - my ;
            suu += u * u ;
            svv += v * v ;
            suv += u * v ;
            suuu += u * u * u ;
            svvv += v * v * v ;
            suuv += u * u * v ;
            suvv += u * v * v ;
        }

        // These equations are demonstrated in paper from R.Bullock (2006)
        float uc = 0.5 * ( suv * ( svvv + suuv ) - svv * ( suuu + suvv ) ) / ( suv * suv - suu * svv ) ;
        float vc = 0.5 * ( suv * ( suuu + suvv ) - suu * ( svvv + suuv ) ) / ( suv * suv - suu * svv ) ;

        // Circle parameters
        setCenter(cv::Point(uc+mx,vc+my)) ;
        setRadius((int)(sqrt(uc*uc+vc*vc+(suu+svv)/rPoints.size()))) ;
    }




} // end of namespace

