import os


def test_moviepy(video_path, audio_path, output_path, fps=25):
    import moviepy.editor as mpe
    
    print('--- moviepy ---')

    video = mpe.VideoFileClip(video_path)
    video = video.set_audio(mpe.AudioFileClip(audio_path))
    video.write_videofile(output_path, fps=fps)


def test_ffmpeg(video_path, audio_path, output_path, fps=25):
    import ffmpeg

    print('--- ffmpeg ---')

    video  = ffmpeg.input(video_path).video # get only video channel
    audio  = ffmpeg.input(audio_path).audio # get only audio channel
    output = ffmpeg.output(video, audio, output_path, vcodec='copy', acodec='aac', strict='experimental')
    ffmpeg.run(output)




root_name = "/data/PIRender_hs/audio_2/video"
out_root = "/data/PIRender_hs/audio_2/demo"
files = os.listdir(root_name)

for file in files:
    if file.endswith(".mp4"):
        video_name = os.path.join(root_name, file)
        audio_name = os.path.join(root_name, file[:-4] + ".wav")
        out_name = os.path.join(out_root, file)
        test_ffmpeg(video_name, audio_name, out_name)
        






