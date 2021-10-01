import cv2
import argparse
import detector

parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--video', help="True/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="Videos/cctv1.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/gun.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

if __name__ == "__main__":
    detect = detector.Detector()

    image = args.image
    video = args.video
    webcam = args.webcam

    if image:
        image_path = args.image_path
        if args.verbose:
            print("Opening "+image_path+" ... ")
        detect.image_detect(image_path)

    if video:
        video_path = args.video_path
        if args.verbose:
            print("Opening "+video_path+" ... ")
        detect.video_detect(video_path)

    if webcam:
        if args.verbose:
            print("--- Opening Camera ---")
        detect.webcam_detect()

    cv2.destroyAllWindows()
