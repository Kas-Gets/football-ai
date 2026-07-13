from utils.video_utils import read_video, save_video
from trackers import Tracker
import cv2

def main():
    # Read video frames from the input video
    video_frames = read_video('input_videos/08fd33_4.mp4') 

    # Initialize the Tracker
    tracker = Tracker(model_path='models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                         read_from_stub=True, 
                                         stub_path='stubs/track_stubs.pkl')

    #save cropped images of players
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        #crop bbox from frame
        cropped_player = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        #save cropped player image
        cv2.imwrite(f'output_videos/player_{track_id}.jpg', cropped_player)

    #Draw output
    ##Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save the video frames to an output video file    
    save_video(output_video_frames, 'output_videos/output.mp4')

if __name__ == "__main__":
    main()