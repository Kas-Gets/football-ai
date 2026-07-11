from utils.video_utils import read_video, save_video
from trackers import Tracker

def main():
    # Read video frames from the input video
    video_frames = read_video('input_videos/08fd33_4.mp4') 

    # Initialize the Tracker
    tracker = Tracker(model_path='models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                         read_from_stub=True, 
                                         stub_path='stubs/track_stubs.pkl')

    #Draw output
    ##Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save the video frames to an output video file    
    save_video(output_video_frames, 'output_videos/output.mp4')

if __name__ == "__main__":
    main()