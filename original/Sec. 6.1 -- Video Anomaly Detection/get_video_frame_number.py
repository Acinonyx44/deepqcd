import os
import cv2


def run():
    cwd = 'K:/DataSets/UCF_Crimes/Videos'

    video_dirs = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
                  'Fighting', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism',
                  'Normal_Videos_event', ]
    video_dirs += ['Testing_Normal_Videos_Anomaly-passwd-yonv1943/Testing_Normal_01',
                   'Testing_Normal_Videos_Anomaly-passwd-yonv1943/Testing_Normal_02',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_01',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_01',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_02',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_03',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_04',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_05',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_06',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_07',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_08',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_09',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_10',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_11',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_12',
                   'Training_Normal_Videos_Anomaly-passwd-yonv1943/Training_Normal_Over1GB', ]

    for video_dir in video_dirs:
        for name in os.listdir(f"{cwd}/{video_dir}"):
            path = f"{cwd}/{video_dir}/{name}"
            frame_number = get_video_info(path)
            print(f"{frame_number:8},    {video_dir}/{name}")


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        frame_number = int(cap.get(7))
    else:
        frame_number = False
    return frame_number


if __name__ == '__main__':
    run()
