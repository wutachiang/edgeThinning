#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

namespace pbcvt {

    using namespace boost::python;
    // using namespace cv;

/**
 * @brief Example function. Basic inner matrix product using explicit matrix conversion.
 * @param left left-hand matrix operand (NdArray required)
 * @param right right-hand matrix operand (NdArray required)
 * @return an NdArray representing the dot-product of the left and right operands
 */
    PyObject *dot(PyObject *left, PyObject *right) {

        cv::Mat leftMat, rightMat;
        leftMat = pbcvt::fromNDArrayToMat(left);
        rightMat = pbcvt::fromNDArrayToMat(right);
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        // Check that the 2-D matrices can be legally multiplied.
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;
        PyObject *ret = pbcvt::fromMatToNDArray(result);
        return ret;
    }
/**
 * @brief Example function. Simply makes a new CV_16UC3 matrix and returns it as a numpy array.
 * @return The resulting numpy array.
 */

	PyObject* makeCV_16UC3Matrix(){
		cv::Mat image = cv::Mat::zeros(240,320, CV_16UC3);
		PyObject* py_image = pbcvt::fromMatToNDArray(image);
		return py_image;
	}

//
/**
 * @brief Example function. Basic inner matrix product using implicit matrix conversion.
 * @details This example uses Mat directly, but we won't need to worry about the conversion in the body of the function.
 * @param leftMat left-hand matrix operand
 * @param rightMat right-hand matrix operand
 * @return an NdArray representing the dot-product of the left and right operands
 */
    cv::Mat dot2(cv::Mat leftMat, cv::Mat rightMat) {
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;

        return result;
    }

    /**
     * \brief Example function. Increments all elements of the given matrix by one.
     * @details This example uses Mat directly, but we won't need to worry about the conversion anywhere at all,
     * it is handled automatically by boost.
     * \param matrix (numpy array) to increment
     * \return
     */
    cv::Mat increment_elements_by_one(cv::Mat matrix){
        matrix += 1.0;
        return matrix;
    }

    void thinningIteration(cv::Mat& img, int iter)
    {
        CV_Assert(img.channels() == 1);
        CV_Assert(img.depth() != sizeof(uchar));
        CV_Assert(img.rows > 3 && img.cols > 3);

        cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

        int nRows = img.rows;
        int nCols = img.cols;

        if (img.isContinuous()) {
            nCols *= nRows;
            nRows = 1;
        }

        int x, y;
        uchar *pAbove;
        uchar *pCurr;
        uchar *pBelow;
        uchar *nw, *no, *ne;    // north (pAbove)
        uchar *we, *me, *ea;
        uchar *sw, *so, *se;    // south (pBelow)

        uchar *pDst;

        // initialize row pointers
        pAbove = NULL;
        pCurr  = img.ptr<uchar>(0);
        pBelow = img.ptr<uchar>(1);

        for (y = 1; y < img.rows-1; ++y) {
            // shift the rows up by one
            pAbove = pCurr;
            pCurr  = pBelow;
            pBelow = img.ptr<uchar>(y+1);

            pDst = marker.ptr<uchar>(y);

            // initialize col pointers
            no = &(pAbove[0]);
            ne = &(pAbove[1]);
            me = &(pCurr[0]);
            ea = &(pCurr[1]);
            so = &(pBelow[0]);
            se = &(pBelow[1]);

            for (x = 1; x < img.cols-1; ++x) {
                // shift col pointers left by one (scan left to right)
                nw = no;
                no = ne;
                ne = &(pAbove[x+1]);
                we = me;
                me = ea;
                ea = &(pCurr[x+1]);
                sw = so;
                so = se;
                se = &(pBelow[x+1]);

                int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) + 
                         (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) + 
                         (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                         (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
                int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
                int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
                int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                    pDst[x] = 1;
            }
        }

        img &= ~marker;
    }


    /**
     * Function for thinning the given binary image
     *
     * Parameters:
     *      src  The source image, binary with range = [0,255]
     *      dst  The destination image
     */
    void thinning(const cv::Mat& src, cv::Mat& dst)
    {
        dst = src.clone();
        dst /= 255;         // convert to binary image

        cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
        cv::Mat diff;

        do {
            thinningIteration(dst, 0);
            thinningIteration(dst, 1);
            cv::absdiff(dst, prev, diff);
            dst.copyTo(prev);
        } 
        while (cv::countNonZero(diff) > 0);

        dst *= 255;
    }

    /**
     * This is an example on how to call the thinning funciton above
     */
    cv::Mat run(cv::Mat src)
    {
    //    auto start = system_clock::now();
        auto start = std::chrono::system_clock::now();
        // cv::Mat src = cv::imread(src_img);
        cv::Mat src_inv;
        // if (!src.data)
        //  return -1;
        cv::threshold(src,src_inv, 200, 255, CV_THRESH_BINARY_INV);
        morphologyEx(src_inv,src_inv,cv::MORPH_OPEN,cv::Mat(5,5,CV_8U),cv::Point(-1,-1),1);
        cv::Mat bw;
        cv::cvtColor(src_inv, bw, CV_BGR2GRAY);
        cv::threshold(bw, bw, 100, 255, CV_THRESH_BINARY);

        thinning(bw, bw);

    //  cv::imshow("src", src);
    //  cv::imshow("dst", bw);
        // cv::imwrite("test.png", bw);
        auto end = std::chrono::system_clock::now();
        double elapsed_seconds = std::chrono::duration_cast<
                std::chrono::duration<double> >(end - start).count();
        // std::cout <<  "consum " << elapsed_seconds << "s" << std::endl;
    //  cv::waitKey();
        // return elapsed_seconds;
        return bw; 
    }


#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
        matFromNDArrayBoostConverter();

        //expose module-level functions
        def("dot", dot);
        def("dot2", dot2);
		def("makeCV_16UC3Matrix", makeCV_16UC3Matrix);

		//from PEP8 (https://www.python.org/dev/peps/pep-0008/?#prescriptive-naming-conventions)
        //"Function names should be lowercase, with words separated by underscores as necessary to improve readability."
        def("increment_elements_by_one", increment_elements_by_one);
        def("thinning", run);
    }

} //end namespace pbcvt
