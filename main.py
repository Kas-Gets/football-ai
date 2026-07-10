from utils.video_utils import read_video, save_video

def main():
    # Read video frames from the input video
    video_frames = read_video('input_videos/08fd33_4.mp4') 

    # Save the video frames to an output video file    
    save_video(video_frames, 'output_videos/output.mp4')

if __name__ == "__main__":
    main()