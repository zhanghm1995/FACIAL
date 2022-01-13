import ffmpeg
import subprocess

audio_name = "slogan-gen"
audio_name = "obama2"
audio_name = "test1"

audio_path = f'../examples/audio_train/{audio_name}.wav'
video_new = f'../examples/test_image/{audio_name}/test_1.avi'
output = f'../examples/test_image/{audio_name}/test_1_audio.avi'
output_mp4 = f'../examples/test_image/{audio_name}/test_1_audio.mp4'

subprocess.call(f"ffmpeg -i {video_new} -i {audio_path} -c copy {output}", shell=True)
subprocess.call(f"ffmpeg -i {output} {output_mp4}", shell=True)