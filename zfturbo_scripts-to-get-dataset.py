# coding: utf-8

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'





def prepare_dl_files():

    out1 = open('download_youtube.bat', 'w')

    file_fake = open('../input/zf-deepfake-dataset/zf_dataset_deepfake.txt', 'r')

    file_real = open('../input/zf-deepfake-dataset/zf_dataset_non_deepfake.txt', 'r')

    youtube_list = []



    while 1:

        line = file_fake.readline().strip()

        if line == '':

            break

        if 'youtube' in line:

            youtube_list.append(line)



    while 1:

        line = file_real.readline().strip()

        if line == '':

            break

        if 'youtube' in line:

            youtube_list.append(line)



    youtube_list = sorted(list(set(youtube_list)))

    print('Videos to download from youtube: {}'.format(len(youtube_list)))

    for i in range(len(youtube_list)):

        you = youtube_list[i]

        out1.write('"../youtube-dl.exe" --download-archive downloaded.txt --id -f "bestvideo[height>=1080,height<=1080,ext=mp4]" {}\n'.format(you))

    out1.close()





def prepare_extractor_files():

    out1 = open('extract_frames.bat', 'w')

    file_fake = open('../input/zf-deepfake-dataset/zf_dataset_deepfake.txt', 'r')

    file_real = open('../input/zf-deepfake-dataset/zf_dataset_non_deepfake.txt', 'r')



    fake_files = []

    while 1:

        line = file_fake.readline().strip()

        if line == '':

            break

        if 'youtube' in line:

            current_id = line.split('=')[-1]

        else:

            arr = line.split('-')

            fn = 'fake/{}_00_{}_00_{}.mp4'.format(current_id, arr[0].replace(':', '_'), arr[1].replace(':', '_'))

            fake_files.append(fn)

            out1.write('ffmpeg -n -i {}.mp4 -ss {} -to {} -c:v libx264 -preset veryslow -crf 15 {}\n'.

                       format(current_id, arr[0], arr[1], fn))

    print('Fake files: {}'.format(len(fake_files)))



    real_files = []

    while 1:

        line = file_real.readline().strip()

        if line == '':

            break

        if 'youtube' in line:

            current_id = line.split('=')[-1]

        else:

            arr = line.split('-')

            fn = 'real/{}_00_{}_00_{}.mp4'.format(current_id, arr[0].replace(':', '_'), arr[1].replace(':', '_'))

            real_files.append(fn)

            out1.write('ffmpeg -n -i {}.mp4 -ss {} -to {} -c:v libx264 -preset veryslow -crf 15 {}\n'.

                       format(current_id, arr[0], arr[1], fn))

    print('Real files: {}'.format(len(real_files)))

    

    out1.close()





prepare_dl_files()

prepare_extractor_files()