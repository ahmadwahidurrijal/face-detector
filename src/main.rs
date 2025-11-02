use opencv::prelude::*;
use opencv::{
    core,        // <-- dibutuhkan untuk AlgorithmHint
    imgproc,
    objdetect,
    videoio,
    Result,
};

const FACE_CASCADE_PATH: &str = "models/haarcascade_frontalface_default.xml";
const EYE_CASCADE_PATH: &str = "models/haarcascade_eye.xml";

fn main() -> Result<()> {
    // 1. Muat classifier Haar Cascade
    let mut face_cascade = objdetect::CascadeClassifier::new(FACE_CASCADE_PATH)?;
    let mut eye_cascade = objdetect::CascadeClassifier::new(EYE_CASCADE_PATH)?;

    // 2. Buka kamera (device 0)
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        panic!("Tidak bisa membuka kamera!");
    }

    println!("Tekan 'q' untuk keluar.");

    let mut frame = core::Mat::default();
    loop {
        // 3. Baca frame
        cam.read(&mut frame)?;
        if frame.empty() {
            break;
        }

        // 4. Konversi ke grayscale (digunakan untuk deteksi)
        let mut gray = core::Mat::default();
        imgproc::cvt_color(
            &frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // 5. Deteksi wajah
        let mut faces = core::Vector::<core::Rect>::new();
        face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,     // scale_factor
            3,       // min_neighbors
            objdetect::CASCADE_SCALE_IMAGE,
            core::Size::new(30, 30), // min_size
            core::Size::new(0, 0),   // max_size
        )?;

        // 6. Untuk setiap wajah, deteksi mata di dalam ROI (Region of Interest)
        for face in faces.iter() {
            // Gambar kotak hijau di sekitar wajah
            imgproc::rectangle(
                &mut frame,
                face,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0), // hijau
                2,
                imgproc::LINE_8,
                0,
            )?;

            // Buat ROI (Region of Interest) dari grayscale dan frame asli
            // Ekstrak koordinat wajah
            let x = face.x;
            let y = face.y;
            let w = face.width;
            let h = face.height;

            // Buat Rect manual (opsional, tapi aman)
            let face_region = core::Rect::new(x, y, w, h);

            let face_gray = gray.roi(face_region)?;
            let mut face_color = frame.roi_mut(face_region)?;

            // Deteksi mata hanya di dalam wajah
            let mut eyes = core::Vector::<core::Rect>::new();
            eye_cascade.detect_multi_scale(
                &face_gray,
                &mut eyes,
                1.1,     // scale_factor
                3,       // min_neighbors
                objdetect::CASCADE_SCALE_IMAGE,
                core::Size::new(5, 5),  // min_size (mata lebih kecil)
                core::Size::new(50, 50), // max_size (opsional, batasi ukuran maks)
            )?;

            // Gambar kotak merah di sekitar mata
            for eye in eyes.iter() {
                imgproc::rectangle(
                    &mut face_color,
                    eye,
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0), // merah (BGR)
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
            }
        }

        // 7. Tampilkan hasil
        opencv::highgui::imshow("Face & Eye Detection (Rust + OpenCV)", &frame)?;

        // 8. Keluar jika tekan 'q'
        if opencv::highgui::wait_key(1)? == 113 { // ASCII 'q'
            break;
        }
    }

    Ok(())
}