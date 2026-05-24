#include <iostream>

#include <string>

#include <vector>

#include <filesystem>

#include <chrono>

#include <iomanip>

#include <sstream>

#include <stdexcept>

#include <windows.h>

#include <cstdlib>

#include <cctype>

#include <algorithm>



#include <opencv2/opencv.hpp>

#include <opencv2/core/utils/logger.hpp>



#include <dlib/dnn.h>

#include <dlib/image_processing.h>

#include <dlib/image_processing/frontal_face_detector.h>

#include <dlib/opencv.h>

#include <thread>

#include <atomic>

#include <mutex>

using namespace std;

using namespace dlib;

namespace fs = std::filesystem;

void drawVideoUI(cv::Mat& frame, const string& status)

{

    // 상단 UI 바

    cv::rectangle(frame,

        cv::Point(0, 0),

        cv::Point(frame.cols, 60),

        cv::Scalar(30, 30, 30),

        cv::FILLED);



    // 버튼 안내

    cv::putText(frame, "[P] Play/Pause", cv::Point(20, 40),

        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);



    cv::putText(frame, "[S] Stop", cv::Point(250, 40),

        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);



    cv::putText(frame, "[ESC] Exit", cv::Point(380, 40),

        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);



    // 상태 표시

    cv::putText(frame,

        "Status: " + status,

        cv::Point(20, frame.rows - 20),

        cv::FONT_HERSHEY_SIMPLEX,

        0.7,

        cv::Scalar(0, 255, 255),

        2);

}

// ---------------- Dlib ResNet Definition ----------------



template <template<int, template<typename>class, int, typename> class block,

    int N,

    template<typename>class BN,

    typename SUBNET>

using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;



template <template<int, template<typename>class, int, typename> class block,

    int N,

    template<typename>class BN,

    typename SUBNET>

using residual_down = add_prev2<

    avg_pool<2, 2, 2, 2,

    skip1<tag2<block<N, BN, 2,

    tag1<SUBNET>>>>> >;



template <int N, template <typename> class BN, int stride, typename SUBNET>

using block = BN<con<N, 3, 3, 1, 1,

    relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;



template <int N, typename SUBNET>

using ares = relu<residual<block, N, affine, SUBNET>>;



template <int N, typename SUBNET>

using ares_down = relu<residual_down<block, N, affine, SUBNET>>;



template <typename SUBNET>

using alevel0 = ares_down<256, SUBNET>;



template <typename SUBNET>

using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;



template <typename SUBNET>

using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;



template <typename SUBNET>

using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;



template <typename SUBNET>

using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;



using anet_type = loss_metric <

    fc_no_bias<128,

    avg_pool_everything<

    alevel0<

    alevel1<

    alevel2<

    alevel3<

    alevel4<

    max_pool<3, 3, 2, 2,

    relu<affine<con<32, 7, 7, 2, 2,

    input_rgb_image_sized<150>

    >>>>>>>>>>>>;



// ---------------- Registered Face Data ----------------



struct FaceRecord

{

    string name;

    matrix<float, 0, 1> descriptor;

};



// ---------------- Face Recognition System Class ----------------



class FaceRecognitionSystem

{

private:

    frontal_face_detector detector_;

    shape_predictor landmarkModel_;

    anet_type recognitionNet_;

    std::vector<FaceRecord> knownFaces_;

    double threshold_;



public:

    FaceRecognitionSystem(

        const string& landmarkPath,

        const string& recognitionModelPath,

        double threshold = 0.75

    )

        : detector_(get_frontal_face_detector()),

        threshold_(threshold)

    {

        cout << "[Model Loading] Landmark model: " << landmarkPath << endl;

        deserialize(landmarkPath) >> landmarkModel_;



        cout << "[Model Loading] Face recognition model: "

            << recognitionModelPath << endl;

        deserialize(recognitionModelPath) >> recognitionNet_;



        cout << "[Model Loaded] All Dlib models loaded successfully." << endl;

    }



    void loadKnownFaces(const string& folderPath)

    {

        cout << "[Known Faces] Folder path: " << folderPath << endl;



        if (!fs::exists(folderPath))

        {

            throw runtime_error("known_faces folder not found: " + folderPath);

        }



        for (const auto& entry : fs::directory_iterator(folderPath))

        {

            if (!entry.is_regular_file())

            {

                continue;

            }



            const string path = entry.path().string();

            const string filename = entry.path().stem().string();

            const string extension = entry.path().extension().string();



            if (!isImageFile(extension))

            {

                cout << "[Skipped] Not an image file: " << path << endl;

                continue;

            }



            string name = extractName(filename);



            cv::Mat image = cv::imread(path);



            if (image.empty())

            {

                cout << "[Warning] Failed to load image: " << path << endl;

                continue;

            }



            auto descriptors = extractDescriptors(image);



            if (descriptors.empty())

            {

                cout << "[Warning] No face detected in image: " << path << endl;

                continue;

            }



            knownFaces_.push_back(FaceRecord{ name, descriptors[0] });



            cout << "[Registered] " << name << " : " << path << endl;

        }



        cout << "[Known Faces] Total registered faces: "

            << knownFaces_.size() << endl;



        if (knownFaces_.empty())

        {

            throw runtime_error("No registered faces found. Check the known_faces folder.");

        }

    }



    std::vector<rectangle> detectFaces(cv::Mat& frame)

    {

        cv_image<bgr_pixel> cimg(frame);

        return detector_(cimg);

    }



    std::vector<matrix<float, 0, 1>> extractDescriptors(cv::Mat& image)

    {

        cv_image<bgr_pixel> cimg(image);

        std::vector<rectangle> faces = detector_(cimg);

        std::vector<matrix<rgb_pixel>> faceChips;



        for (const auto& face : faces)

        {

            auto shape = landmarkModel_(cimg, face);



            matrix<rgb_pixel> faceChip;



            extract_image_chip(

                cimg,

                get_face_chip_details(shape, 150, 0.25),

                faceChip

            );



            faceChips.push_back(std::move(faceChip));

        }



        if (faceChips.empty())

        {

            return {};

        }



        return recognitionNet_(faceChips);

    }



    string recognize(const matrix<float, 0, 1>& descriptor, double& bestDistance)

    {

        string bestName = "Human";

        bestDistance = 999.0;



        for (const auto& known : knownFaces_)

        {

            double distance = length(descriptor - known.descriptor);



            if (distance < bestDistance)

            {

                bestDistance = distance;

                bestName = known.name;

            }

        }



        if (bestDistance > threshold_)

        {

            return "Unknown";

        }



        return bestName;

    }



    void processFrame(cv::Mat& frame)

    {

        cv_image<bgr_pixel> cimg(frame);

        std::vector<rectangle> faces = detector_(cimg);



        std::vector<matrix<rgb_pixel>> faceChips;

        std::vector<rectangle> validFaces;



        for (const auto& face : faces)

        {

            auto shape = landmarkModel_(cimg, face);



            matrix<rgb_pixel> faceChip;



            extract_image_chip(

                cimg,

                get_face_chip_details(shape, 150, 0.25),

                faceChip

            );



            faceChips.push_back(std::move(faceChip));

            validFaces.push_back(face);

        }



        if (faceChips.empty())

        {

            return;

        }



        std::vector<matrix<float, 0, 1>> descriptors = recognitionNet_(faceChips);



        for (size_t i = 0; i < descriptors.size(); i++)

        {

            double distance = 0.0;

            string name = recognize(descriptors[i], distance);



            drawResult(frame, validFaces[i], name, distance);

        }

    }



private:

    bool isImageFile(const string& extension)

    {

        string ext = extension;



        for (auto& c : ext)

        {

            c = static_cast<char>(tolower(static_cast<unsigned char>(c)));

        }



        return ext == ".jpg" ||

            ext == ".jpeg" ||

            ext == ".png" ||

            ext == ".bmp";

    }



    string extractName(const string& filename)

    {

        size_t pos = filename.find('_');



        if (pos == string::npos)

        {

            return filename;

        }



        return filename.substr(0, pos);

    }



    void drawResult(

        cv::Mat& frame,

        const rectangle& face,

        const string& name,

        double distance

    )

    {

        cv::Scalar color;



        if (name == "Unknown")

        {

            color = cv::Scalar(0, 0, 255);

        }

        else

        {

            color = cv::Scalar(0, 255, 0);

        }



        cv::Rect rect(

            static_cast<int>(face.left()),

            static_cast<int>(face.top()),

            static_cast<int>(face.width()),

            static_cast<int>(face.height())

        );



        cv::rectangle(frame, rect, color, 2);



        stringstream ss;

        ss << name << " ";

        ss << fixed << setprecision(2) << distance;



        int textY = std::max(20, static_cast<int>(face.top()) - 10);



        cv::putText(

            frame,

            ss.str(),

            cv::Point(static_cast<int>(face.left()), textY),

            cv::FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2

        );

    }

};

std::mutex frameMutex;

cv::Mat sharedFrame;

cv::Mat resultFrame;



std::atomic<bool> running(true);

std::atomic<bool> paused(false);

// ---------------- FPS Counter Class ----------------



class FPSCounter

{

private:

    chrono::steady_clock::time_point lastTime_;

    double fps_;



public:

    FPSCounter()

        : lastTime_(chrono::steady_clock::now()),

        fps_(0.0)

    {

    }



    double update()

    {

        auto currentTime = chrono::steady_clock::now();

        chrono::duration<double> elapsed = currentTime - lastTime_;



        if (elapsed.count() > 0.0)

        {

            fps_ = 1.0 / elapsed.count();

        }



        lastTime_ = currentTime;



        return fps_;

    }

};

void faceWorker(FaceRecognitionSystem& faceSystem)
{
    while (running)
    {
        if (paused)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        cv::Mat localFrame;

        // 공유 프레임 복사
        {
            std::lock_guard<std::mutex> lock(frameMutex);

            if (sharedFrame.empty())
                continue;

            localFrame = sharedFrame.clone();
        }

        // =========================
        // 속도 최적화용 축소 프레임
        // =========================

        cv::Mat smallFrame;

        cv::resize(
            localFrame,
            smallFrame,
            cv::Size(),
            0.5,
            0.5
        );

        // 축소된 프레임으로 얼굴 인식 수행
        faceSystem.processFrame(smallFrame);

        // =========================
        // 결과 저장
        // =========================
        {
            std::lock_guard<std::mutex> lock(frameMutex);

            // 현재는 축소된 화면 저장
            // (속도 우선)
            sharedFrame = smallFrame.clone();
        }

        // CPU 점유율 조절
        std::this_thread::sleep_for(
            std::chrono::milliseconds(1)
        );
    }
}

// ---------------- Main Function ----------------



int main()

{

    _putenv_s("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0");



    SetConsoleOutputCP(CP_UTF8);

    SetConsoleCP(CP_UTF8);



    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);



    try

    {

        const string basePath = "C:/Users/skt16/Desktop/Project3/x64/Debug/";



        FaceRecognitionSystem faceSystem(

            basePath + "shape_predictor_5_face_landmarks.dat",

            basePath + "dlib_face_recognition_resnet_model_v1.dat",

            0.6

        );



        faceSystem.loadKnownFaces(basePath + "known_faces");



        FPSCounter fpsCounter;



        while (true)

        {

            cout << "\n=====================================\n";

            cout << "     FACE RECOGNITION SYSTEM\n";

            cout << "=====================================\n";

            cout << "[1] Run Image Folder\n";

            cout << "[2] Run Video (MP4)\n";

            cout << "[0] Exit\n";

            cout << "-------------------------------------\n";

            cout << "Select option: ";



            int menu;

            cin >> menu;



            if (menu == 0)

                break;



            string path;



            // =========================

            // 1. 이미지 폴더

            // =========================

            if (menu == 1)

            {

                cout << "Enter image folder path: ";

                cin >> path;



                if (!fs::exists(path))

                {

                    cout << "[Error] Folder not found\n";

                    continue;

                }



                std::vector<string> images;



                for (auto& entry : fs::directory_iterator(path))

                {

                    string ext = entry.path().extension().string();

                    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);



                    if (ext == ".jpg" || ext == ".jpeg" ||

                        ext == ".png" || ext == ".bmp")

                    {

                        images.push_back(entry.path().string());

                    }

                }



                sort(images.begin(), images.end());



                for (const auto& imgPath : images)

                {

                    cv::Mat frame = cv::imread(imgPath);



                    if (frame.empty())

                    {

                        cout << "[Skip] " << imgPath << endl;

                        continue;

                    }



                    faceSystem.processFrame(frame);



                    double fps = fpsCounter.update();



                    cv::putText(frame,

                        "FPS: " + to_string((int)fps),

                        cv::Point(10, 30),

                        cv::FONT_HERSHEY_SIMPLEX,

                        0.8,

                        cv::Scalar(255, 255, 0),

                        2

                    );



                    cv::imshow("Image Mode", frame);

                    int key = cv::waitKey(700);

                    if (key == 27)
                        break;

                }

            }



            // =========================

            // 2. 영상

            // =========================

            else if (menu == 2)

            {

                cout << "Enter video file path: ";

                cin >> path;



                cv::VideoCapture cap(path);

                if (!cap.isOpened()) {

                    cout << "[Error] Failed to open video file\n";

                    continue;

                }



                running = true;

                paused = false;



                // 스레드 시작

                std::thread worker(faceWorker, std::ref(faceSystem));



                cv::Mat frame;

                string status = "Playing";



                while (true)

                {

                    // 1. 영상 읽기

                    if (!paused) {

                        if (!cap.read(frame)) {

                            running = false;

                            break;

                        }

                        // 원본 프레임을 스레드로 전달

                        std::lock_guard<std::mutex> lock(frameMutex);

                        sharedFrame = frame.clone();

                    }



                    // 2. 결과 렌더링

                    cv::Mat render;

                    {

                        std::lock_guard<std::mutex> lock(frameMutex);

                        if (!sharedFrame.empty())

                            render = sharedFrame.clone();

                    }



                    if (!render.empty()) {

                        drawVideoUI(render, status);

                        cv::imshow("AI Video Player", render);

                    }



                    // 3. 입력 처리

                    int key = cv::waitKey(1);

                    if (key == 27) { // ESC

                        running = false;

                        break;

                    }

                    else if (key == 'p' || key == 'P') {

                        paused = !paused;

                        status = paused ? "Paused" : "Playing";

                    }

                }



                running = false;

                if (worker.joinable()) worker.join();

                cap.release();

            }

        }



        cv::destroyAllWindows();

        cout << "[Program End]" << endl;

    }

    catch (const exception& e)

    {

        cout << "[Exception] " << e.what() << endl;

    }



    return 0;

}
