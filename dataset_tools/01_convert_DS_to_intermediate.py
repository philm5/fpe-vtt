import os
import glob
import json
from abc import abstractmethod
from pathlib import Path
import re
from tqdm import tqdm
from urllib.parse import urlparse
import numpy as np
import subprocess
import pandas as pd
import xml.etree.ElementTree as ElementTree
import math

class PrepareVTTData:
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir):
        self.vtt_datasets_base_path = vtt_datasets_base_path
        self.datasplit = datasplit
        self.raw_video_path = raw_video_path
        self.raw_captions_dir = raw_captions_dir
        self.dst_path = os.path.join(self.vtt_datasets_base_path, self.datasplit)

        Path(self.dst_path).mkdir(parents=True, exist_ok=True)

    def create_multi_proc_script(self, cmds, tgt_dir, cmd_name, num_splits=128):

        res = np.asarray(cmds)
        splits = np.array_split(res, num_splits)
        for split_idx, split in enumerate(splits):
            with open(os.path.join(tgt_dir, f"{cmd_name}_{split_idx:05d}_partition.sh"), 'w') as f:
                f.writelines("\n".join(list(split)))

        with open(os.path.join(tgt_dir, f"{cmd_name}_nohup_ALL_partition.sh"), 'w') as f:
            for i in range(num_splits):
                f.write(f"nohup bash {cmd_name}_{i:05d}_partition.sh >log_{cmd_name}_{i}.txt 2>&1 &\n")

    def video2frame_audio(self):
        #, video_path, datasplit, dst_path
        video_wcard = os.path.join(self.raw_video_path, '*')
        videos = glob.glob(video_wcard)
        frames_path = os.path.join(self.dst_path, 'frames')
        audio_path = os.path.join(self.dst_path, 'audio')

        extract_frame_commands = [f"mkdir -p {frames_path}"]
        extract_audio_commands = [f"mkdir -p {audio_path}"]

        video_gt = []

        video_id_rgx = re.compile('\D*(\d+)')

        for video in videos:
            v_id_str = os.path.splitext(os.path.basename(video))[0]
            m = video_id_rgx.match(v_id_str)
            if m:
                video_id_int = int(m.group(1))
                video_id = f"video_{self.datasplit}_{video_id_int:05d}"

                frames_dst_path = os.path.join(frames_path, video_id)
                mkdir_frame_cmd = f"mkdir {frames_dst_path}"
                ffmpeg_cmd = f"ffmpeg -i {video} '{frames_dst_path}/%d.jpg'"
                ffmpeg_audio_cmd = f"ffmpeg -i {video} '{audio_path}/{video_id}.wav'"
                extract_frame_commands.append(" && ".join([mkdir_frame_cmd, ffmpeg_cmd]))  # , rm_cmd, rm_dir ]))
                extract_audio_commands.append(ffmpeg_audio_cmd)  # , rm_cmd, rm_dir ]))

                single_gt = {
                    'raw_video_path': video,
                    'video_id': video_id,
                    'src_id': v_id_str,
                    'src_id_int': video_id_int
                }
                video_gt.append(single_gt)


        tgt_dir = os.path.join(self.dst_path, 'scripts')
        os.makedirs(tgt_dir, exist_ok=True)
        self.create_multi_proc_script(cmds=extract_frame_commands,
                                      tgt_dir=tgt_dir,
                                      cmd_name='extract_frame')
        self.create_multi_proc_script(cmds=extract_audio_commands,
                                      tgt_dir=tgt_dir,
                                      cmd_name='extract_audio')

        # ex_frame_path = os.path.join(self.dst_path, "extract_frame.sh")
        # ex_audio_path = os.path.join(self.dst_path, "extract_audio.sh")
        #
        # with open(ex_frame_path, 'w') as fout:
        #     fout.write("\n".join(extract_frame_commands))
        # os.chmod(ex_frame_path, 0o755)
        # with open(ex_audio_path, 'w') as fout:
        #     fout.write("\n".join(extract_audio_commands))
        # os.chmod(ex_audio_path, 0o755)

        return video_gt

    def create_dataset_info(self):
        self.video_gt = self.create_frame_audio_extract_scripts()
        sentences = self.create_sentences_gt()
        dataset_info = {
            'info': {
                'dataset_path': self.dst_path,
                'name': self.datasplit
            },
            'sentences': sentences,
            'videos': self.video_gt
        }

        with open(os.path.join(self.dst_path, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f)

    @abstractmethod
    def create_frame_audio_extract_scripts(self):
        pass

    #@abstractmethod
    def create_sentences_gt(self):
        return []

    def get_caption_gt(self, old_gt, caption):
        gt_with_caption = {
            'video_id': f"video_{self.datasplit}_{old_gt['video_id']}",
            'caption': caption
        }

        if 'ytVideoID' in old_gt:
            gt_with_caption['ytVideoID'] = old_gt['ytVideoID']

        return gt_with_caption

class PrepareTrecvidData(PrepareVTTData):
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir):
        super().__init__(vtt_datasets_base_path, datasplit, raw_video_path, raw_captions_dir)

    def video2frame_audio(self):
        #, video_path, datasplit, dst_path
        video_wcard = os.path.join(self.raw_video_path, '*')
        videos = glob.glob(video_wcard)
        frames_path = os.path.join(self.dst_path, 'frames')
        audio_path = os.path.join(self.dst_path, 'audio')

        extract_frame_commands = [f"mkdir -p {frames_path}"]
        extract_audio_commands = [f"mkdir -p {audio_path}"]

        video_gt = []

        video_id_rgx = re.compile('\D*(\d+)')

        for video in videos:
            v_id_str = os.path.splitext(os.path.basename(video))[0]
            m = video_id_rgx.match(v_id_str)
            if m:
                video_id_int = int(m.group(1))
                video_id = f"video_{self.datasplit}_{video_id_int:05d}"

                frames_dst_path = os.path.join(frames_path, video_id)
                mkdir_frame_cmd = f"mkdir {frames_dst_path}"
                ffmpeg_cmd = f"ffmpeg -i {video} '{frames_dst_path}/%d.jpg'"
                ffmpeg_audio_cmd = f"ffmpeg -i {video} '{audio_path}/{video_id}.wav'"
                extract_frame_commands.append(" && ".join([mkdir_frame_cmd, ffmpeg_cmd]))  # , rm_cmd, rm_dir ]))
                extract_audio_commands.append(ffmpeg_audio_cmd)  # , rm_cmd, rm_dir ]))

                single_gt = {
                    'raw_video_path': video,
                    'video_id': video_id,
                    'src_id': v_id_str,
                    'src_id_int': video_id_int
                }
                video_gt.append(single_gt)

        ex_frame_path = os.path.join(self.dst_path, "extract_frame.sh")
        ex_audio_path = os.path.join(self.dst_path, "extract_audio.sh")

        with open(ex_frame_path, 'w') as fout:
            fout.write("\n".join(extract_frame_commands))
        os.chmod(ex_frame_path, 0o755)
        with open(ex_audio_path, 'w') as fout:
            fout.write("\n".join(extract_audio_commands))
        os.chmod(ex_audio_path, 0o755)

        return video_gt

    def create_frame_audio_extract_scripts(self):
        return self.video2frame_audio()

    def get_video_sentence_mapping(self, vtt_gt_file):
        with open(vtt_gt_file, 'r') as f:
            flines = f.read()

        video_sentence_mapping = {}  # [line.split(' ') for line in flines.split('\n')]
        for line in flines.split('\n'):
            split_line = line.split(' ')
            if len(split_line) == 3:
                vsm_entry = {
                    'src_id': split_line[0],
                    'cap_id_a': split_line[1],
                    'cap_id_b': split_line[2]
                }

                video_sentence_mapping[split_line[0]] = vsm_entry

        return video_sentence_mapping

    def get_captions_mapping(self, captions_file):
        with open(captions_file, 'r', encoding='latin1') as f:
            flines = f.read()

        captions_mapping = {}
        for ln_no, line in enumerate(flines.split('\n')):
            split_line = line.split('    ')
            if len(split_line) == 2:
                captions_mapping[split_line[0]] = split_line[1]
            else:
                print(f'error with line #{ln_no}: {line}')

        return captions_mapping

    def create_sentences_gt(self):
        vtt_gt_file = os.path.join(self.raw_captions_dir, 'vtt.gt')
        testing_set_a = os.path.join(self.raw_captions_dir, 'vines.textDescription.A.testingSet')
        testing_set_b = os.path.join(self.raw_captions_dir, 'vines.textDescription.B.testingSet')

        vsm = self.get_video_sentence_mapping(vtt_gt_file)
        cap_mapping_a = self.get_captions_mapping(testing_set_a)
        cap_mapping_b = self.get_captions_mapping(testing_set_b)

        new_gt = []
        for single_gt in self.video_gt:
            mapping = vsm[single_gt['src_id']]
            cap_a = cap_mapping_a[mapping['cap_id_a']]
            cap_b = cap_mapping_b[mapping['cap_id_b']]
            new_gt.append(self.get_caption_gt(single_gt, cap_a))
            new_gt.append(self.get_caption_gt(single_gt, cap_b))

        return new_gt

class PrepareTrecvidData2020(PrepareVTTData):
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir):
        super().__init__(vtt_datasets_base_path, datasplit, raw_video_path, raw_captions_dir)

    def create_frame_audio_extract_scripts(self):
        return self.video2frame_audio()

    def get_captions_mapping(self, captions_file):
        rgx = re.compile('(\d+) (.*)')
        with open(captions_file, 'r', encoding='latin1') as f:
            flines = f.read()

        captions_mapping = {}
        for ln_no, line in enumerate(flines.split('\n')):
            m = rgx.match(line)
            if m:
                video_id = m.group(1)
                if video_id in captions_mapping:
                    captions_mapping[video_id].append(m.group(2))
                else:
                    captions_mapping[video_id] = [m.group(2)]
            else:
                print(f'error with line #{ln_no}: {line}')

        return captions_mapping

    def create_sentences_gt(self):
        vtt_gt_file = os.path.join(self.raw_captions_dir, 'vtt_ground_truth.txt')
        cap_mapping = self.get_captions_mapping(vtt_gt_file)
        # cap_mapping_b = self.get_captions_mapping(testing_set_b)
        #
        new_gt = []
        for single_gt in self.video_gt:
            video_id = single_gt['src_id']
            single_vid_caps = cap_mapping[video_id]
            for single_cap in single_vid_caps:
                new_gt.append(self.get_caption_gt(single_gt, single_cap))

        return new_gt

class PrepareTrecvidTestData2020(PrepareVTTData):
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir):
        super().__init__(vtt_datasets_base_path, datasplit, raw_video_path, raw_captions_dir)

    def create_frame_audio_extract_scripts(self):
        return self.video2frame_audio()

    def get_captions_mapping(self, captions_files):
        rgx = re.compile('(\d+) (.*)')
        flines = ''
        for captions_file in captions_files:
            with open(captions_file, 'r', encoding='latin1') as f:
                single_file_lines = f.read()
                flines += single_file_lines

        captions_mapping = {}
        splitted = flines.split('\n')
        for ln_no, line in enumerate(splitted):
            m = rgx.match(line)
            if m:
                video_id = m.group(1)
                if video_id in captions_mapping:
                    captions_mapping[video_id].append(m.group(2))
                else:
                    captions_mapping[video_id] = [m.group(2)]
            else:
                print(f'error with line #{ln_no}/{len(splitted)}: {line}')

        return captions_mapping

    def create_sentences_gt(self):
        files = glob.glob(os.path.join(self.raw_captions_dir, 'MR_Set_*.txt'))
        cap_mapping = self.get_captions_mapping(files)
        # cap_mapping_b = self.get_captions_mapping(testing_set_b)
        #
        new_gt = []
        for single_gt in self.video_gt:
            video_id = single_gt['src_id']
            single_vid_caps = cap_mapping[video_id]
            for single_cap in single_vid_caps:
                new_gt.append(self.get_caption_gt(single_gt, single_cap))

        return new_gt

class PrepareTrecvidTestData2021(PrepareVTTData):
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir):
        super().__init__(vtt_datasets_base_path, datasplit, raw_video_path, raw_captions_dir)

    def create_frame_audio_extract_scripts(self):
        return self.video2frame_audio()

    def get_captions_mapping(self, captions_files):
        rgx = re.compile('(\d+) (.*)')
        flines = ''
        for captions_file in captions_files:
            with open(captions_file, 'r', encoding='latin1') as f:
                single_file_lines = f.read()
                flines += single_file_lines

        captions_mapping = {}
        splitted = flines.split('\n')
        for ln_no, line in enumerate(splitted):
            m = rgx.match(line)
            if m:
                video_id = m.group(1)
                if video_id in captions_mapping:
                    captions_mapping[video_id].append(m.group(2))
                else:
                    captions_mapping[video_id] = [m.group(2)]
            else:
                print(f'error with line #{ln_no}/{len(splitted)}: {line}')

        return captions_mapping

    def create_sentences_gt(self):
        #files = glob.glob(os.path.join(self.raw_captions_dir, 'MR_Set_*.txt'))
        #cap_mapping = self.get_captions_mapping(files)
        # cap_mapping_b = self.get_captions_mapping(testing_set_b)
        #
        new_gt = []
        for single_gt in self.video_gt:
            video_id = single_gt['src_id']
            single_vid_caps = [""] #cap_mapping[video_id]
            for single_cap in single_vid_caps:
                new_gt.append(self.get_caption_gt(single_gt, single_cap))

        return new_gt

class PrepareMSRData(PrepareVTTData):
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir):
        super().__init__(vtt_datasets_base_path, datasplit, raw_video_path, raw_captions_dir)


    def video2frame_audio(self):

        vtt_gt_file = os.path.join(self.raw_captions_dir, 'videodatainfo_2017.json')

        with open(vtt_gt_file, 'r') as fin:
            d = json.load(fin)

        videos_videodatainfo = d['videos']

        #, video_path, datasplit, dst_path
        video_wcard = os.path.join(self.raw_video_path, '*')
        videos = glob.glob(video_wcard)
        frames_path = os.path.join(self.dst_path, 'frames')
        audio_path = os.path.join(self.dst_path, 'audio')

        extract_frame_commands = [f"mkdir -p {frames_path}"]
        extract_audio_commands = [f"mkdir -p {audio_path}"]

        video_gt = []

        video_id_rgx = re.compile('\D*(\d+)')

        for video in videos:
            v_id_str = os.path.splitext(os.path.basename(video))[0]
            m = video_id_rgx.match(v_id_str)


            # match with vdinfo
            a = 5
            if m:


                video_id_int = int(m.group(1))

                videodatainfo = [x for x in videos_videodatainfo if x['video_id'] == f'video{video_id_int}']

                if len(videodatainfo) == 1:
                    videodatainfo = videodatainfo[0]
                else:
                    print(f"Error while finding videodatainfo for 'video{video_id_int}'")
                    break

                video_id = f"video_{self.datasplit}_{video_id_int:05d}"

                frames_dst_path = os.path.join(frames_path, video_id)
                mkdir_frame_cmd = f"mkdir {frames_dst_path}"

                ffmpeg_cmd = f"ffmpeg -i {video} -ss {videodatainfo['start time']} -t {videodatainfo['end time'] - videodatainfo['start time']} '{frames_dst_path}/%d.jpg'"
                ffmpeg_audio_cmd = f"ffmpeg -i {video} -ss  {videodatainfo['start time']} -t {videodatainfo['end time'] - videodatainfo['start time']} -q:a 0 '{audio_path}/{video_id}.wav'"

                # ffmpeg_cmd = f"ffmpeg -i {video} '{frames_dst_path}/%d.jpg'"
                # ffmpeg_audio_cmd = f"ffmpeg -i {video} '{audio_path}/{video_id}.wav'"
                extract_frame_commands.append(" && ".join([mkdir_frame_cmd, ffmpeg_cmd]))  # , rm_cmd, rm_dir ]))
                extract_audio_commands.append(ffmpeg_audio_cmd)  # , rm_cmd, rm_dir ]))

                single_gt = {
                    'raw_video_path': video,
                    'video_id': video_id,
                    'src_id': v_id_str,
                    'src_id_int': video_id_int
                }
                video_gt.append(single_gt)

        ex_frame_path = os.path.join(self.dst_path, "extract_frame.sh")
        ex_audio_path = os.path.join(self.dst_path, "extract_audio.sh")

        with open(ex_frame_path, 'w') as fout:
            fout.write("\n".join(extract_frame_commands))
        os.chmod(ex_frame_path, 0o755)
        with open(ex_audio_path, 'w') as fout:
            fout.write("\n".join(extract_audio_commands))
        os.chmod(ex_audio_path, 0o755)

        return video_gt

    def create_frame_audio_extract_scripts(self):
        return self.video2frame_audio()

    def create_sentences_gt(self):
        vtt_gt_file = os.path.join(self.raw_captions_dir, 'videodatainfo_2017.json')
        with open(vtt_gt_file, 'r') as f:
            vtt_gt = json.load(f)

        sentences = vtt_gt['sentences']

        video_id_rgx = re.compile('\D*(\d+)')
        vid_to_sentences_dict = {}
        for sent in sentences:
            m = video_id_rgx.match(sent['video_id'])
            if m:
                video_id = int(m.group(1))
                if not video_id in vid_to_sentences_dict:
                    vid_to_sentences_dict[video_id] = []

                vid_to_sentences_dict[video_id].append(sent)

        new_gt = []
        for single_gt in self.video_gt:
            for cap in vid_to_sentences_dict[single_gt['src_id_int']]:
                new_gt.append(self.get_caption_gt(single_gt, cap['caption']))

        return new_gt


class PrepareAcmGifData(PrepareVTTData):
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir):
        super().__init__(vtt_datasets_base_path, datasplit, raw_video_path, raw_captions_dir)


    def video2frame_audio(self):

        vtt_gt_file = os.path.join(self.raw_captions_dir, 'pre-training.json')

        with open(vtt_gt_file, 'r') as f:
            pre_training = json.load(f)

        existing_files = []
        non_existing_files = []
        for sample in tqdm(pre_training):
            url = sample['url']
            disassembled = urlparse(url)
            ext = os.path.splitext(disassembled.path)[1]
            local_path = os.path.join(self.raw_video_path, f"{sample['id']}{ext}")
            sample['local_path'] = local_path
            if os.path.exists(local_path):
                existing_files.append(sample)
            else:
                if not (ext == '.gif' or ext == '.gifv'):
                    found = False
                    for ext in ['.gif', '.gifv']:
                        local_path = os.path.join(self.raw_video_path, f"{sample['id']}{ext}")
                        if os.path.exists(local_path):
                            found = True
                            sample['local_path'] = local_path
                            existing_files.append(sample)
                            break
                    if not found:
                        non_existing_files.append(sample)
                else:
                    non_existing_files.append(sample)


        #, video_path, datasplit, dst_path
        frames_path = os.path.join(self.dst_path, 'frames')
        audio_path = os.path.join(self.dst_path, 'audio')

        extract_frame_commands = [f"mkdir -p {frames_path}"]
        extract_audio_commands = [f"mkdir -p {audio_path}"]

        video_gt = []

        video_id_rgx = re.compile('\D*(\d+)')

        for single_file in existing_files:
            video_path = single_file['local_path']
            v_id_str = os.path.splitext(os.path.basename(video_path))[0]
            m = video_id_rgx.match(v_id_str)


            # match with vdinfo
            a = 5
            if m:


                video_id_int = int(m.group(1))
                video_id = f"video_{self.datasplit}_{video_id_int:05d}"

                frames_dst_path = os.path.join(frames_path, video_id)
                mkdir_frame_cmd = f"mkdir {frames_dst_path}"

                ffmpeg_cmd = f"ffmpeg -f gif -i {video_path} '{frames_dst_path}/%d.jpg'"
                ffmpeg_audio_cmd = f"ffmpeg -i -q:a 0 '{audio_path}/{video_id}.wav'"

                # ffmpeg_cmd = f"ffmpeg -i {video} '{frames_dst_path}/%d.jpg'"
                # ffmpeg_audio_cmd = f"ffmpeg -i {video} '{audio_path}/{video_id}.wav'"
                extract_frame_commands.append(" && ".join([mkdir_frame_cmd, ffmpeg_cmd]))  # , rm_cmd, rm_dir ]))
                extract_audio_commands.append(ffmpeg_audio_cmd)  # , rm_cmd, rm_dir ]))

                single_gt = {
                    'raw_video_path': video_path,
                    'video_id': video_id,
                    'src_id': v_id_str,
                    'src_id_int': video_id_int
                }
                video_gt.append(single_gt)

        ex_frame_path = os.path.join(self.dst_path, "extract_frame.sh")
        ex_audio_path = os.path.join(self.dst_path, "extract_audio.sh")

        with open(ex_frame_path, 'w') as fout:
            fout.write("\n".join(extract_frame_commands))
        os.chmod(ex_frame_path, 0o755)
        with open(ex_audio_path, 'w') as fout:
            fout.write("\n".join(extract_audio_commands))
        os.chmod(ex_audio_path, 0o755)

        return video_gt

    def create_frame_audio_extract_scripts(self):
        return self.video2frame_audio()

    def create_sentences_gt(self):
        vtt_gt_file = os.path.join(self.raw_captions_dir, 'pre-training.json')
        with open(vtt_gt_file, 'r') as f:
            vtt_gt = json.load(f)

        sentences = vtt_gt

        video_id_rgx = re.compile('\D*(\d+)')
        vid_to_sentences_dict = {}
        for vtt_single_gt in vtt_gt:
            m = video_id_rgx.match(vtt_single_gt['id'])
            if m:
                video_id = int(m.group(1))
                if not video_id in vid_to_sentences_dict:
                    vid_to_sentences_dict[video_id] = []

                vid_to_sentences_dict[video_id].extend(vtt_single_gt['sentences'])

        new_gt = []
        for single_gt in self.video_gt:
            for cap in vid_to_sentences_dict[single_gt['src_id_int']]:
                new_gt.append(self.get_caption_gt(single_gt, cap))

        return new_gt

class PrepareVatexData(PrepareVTTData):
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir,
                 gt_filename,
                 constant_fps=None):
        super().__init__(vtt_datasets_base_path, datasplit, raw_video_path, raw_captions_dir)
        self.gt_filename = gt_filename
        self.constant_fps = constant_fps

    def ffprobe(self, executable, filename):
        '''Runs ``ffprobe`` executable over ``filename``, returns parsed XML

        Parameters:

            executable (str): Full path leading to ``ffprobe``
            filename (str): Full path leading to the file to be probed

        Returns:

            xml.etree.ElementTree: containing all parsed elements

        '''

        cmd = [
            executable,
            '-v', 'quiet',
            '-print_format', 'xml',  # here is the trick
            '-show_format',
            '-show_streams',
            filename,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE)

        # print(result.stdout.decode('utf-8'))
        return ElementTree.fromstring(result.stdout.decode('utf-8'))

    def video2frame_audio(self):

        vtt_gt_file = os.path.join(self.raw_captions_dir, self.gt_filename)

        with open(vtt_gt_file, 'r') as f:
            vatex = json.load(f)


        video_files = os.listdir(self.raw_video_path)
        streams_per_video_id = self.identify_video_audio_streams_from_raw_files()

        not_found = []


        # split video ids and start/end time
        rgx = re.compile("(.*)_(.*)_(.*)")

        vatex_new = []
        for sample in tqdm(vatex):
            raw_vid = sample['videoID']
            m = rgx.match(raw_vid)
            if m:
                new_sample = sample.copy()
                new_sample['ytVideoID'] = m.groups()[0]
                new_sample['startTime'] = int(m.groups()[1])
                new_sample['endTime'] = int(m.groups()[2])

                vid = new_sample['ytVideoID']
                if vid in streams_per_video_id:
                    video_audio_files = streams_per_video_id[vid]
                    # print(video_audio_files)
                    if 'video' in video_audio_files:
                        new_sample['raw_video_path'] = os.path.join(self.raw_video_path, video_audio_files['video'][0])
                    else:
                        new_sample['raw_video_path'] = None

                    if 'audio' in video_audio_files:
                        new_sample['raw_audio_path'] = os.path.join(self.raw_video_path, video_audio_files['audio'][0])
                    else:
                        new_sample['raw_audio_path'] = None
                else:
                    new_sample['raw_audio_path'] = None
                    new_sample['raw_video_path'] = None
                vatex_new.append(new_sample)
                # Old code....
                # rexp = f".*{re.escape(vid)}.*mp4"
                # rgx3 = re.compile(rexp)
                # results = list(filter(rgx3.match, video_files))
                # if len(results) == 1:
                #     new_sample['raw_video_path'] = os.path.join(self.raw_video_path, results[0])
                # else:
                #     rexp = f".*{re.escape(vid)}.*webm"
                #     rgx2 = re.compile(rexp)
                #     results = list(filter(rgx2.match, video_files))
                #     if len(results) == 1:
                #         new_sample['raw_video_path'] = os.path.join(self.raw_video_path, results[0])
                #     else:
                #         new_sample['raw_video_path'] = None
                #         not_found.append(new_sample['videoID'])
                #
                # vatex_new.append(new_sample)
            else:
                # print(raw_vid)
                pass

        self.vtt_gt = vatex_new

        frames_path = os.path.join(self.dst_path, 'frames')
        audio_path = os.path.join(self.dst_path, 'audio')

        extract_frame_commands = [f"mkdir -p {frames_path}"]
        extract_audio_commands = [f"mkdir -p {audio_path}"]

        video_gt = []

        for sample in self.vtt_gt:
            if sample['raw_video_path'] != None:
                video = sample['raw_video_path']
                audio = sample['raw_audio_path']

                video_id = f"video_{self.datasplit}_{sample['videoID']}"

                frames_dst_path = os.path.join(frames_path, video_id)
                mkdir_frame_cmd = f"mkdir {frames_dst_path}"

                if self.constant_fps is not None:
                    fpscmd = f' -vf fps={self.constant_fps}'
                else:
                    fpscmd = ''

                ffmpeg_cmd = f"ffmpeg -i {video} -ss {sample['startTime']} -t {sample['endTime'] - sample['startTime']}{fpscmd} '{frames_dst_path}/%d.jpg'"
                ffmpeg_audio_cmd = f"ffmpeg -i {audio} -ss {sample['startTime']} -t {sample['endTime'] - sample['startTime']} -q:a 0 '{audio_path}/{video_id}.wav'"

                # ffmpeg_cmd = f"ffmpeg -i {video} '{frames_dst_path}/%d.jpg'"
                # ffmpeg_audio_cmd = f"ffmpeg -i {video} '{audio_path}/{video_id}.wav'"
                extract_frame_commands.append(" && ".join([mkdir_frame_cmd, ffmpeg_cmd]))  # , rm_cmd, rm_dir ]))
                extract_audio_commands.append(ffmpeg_audio_cmd)  # , rm_cmd, rm_dir ]))

                single_gt = {
                    'raw_video_path': video,
                    'video_id': video_id,
                    'src_id': sample['videoID'],
                }
                video_gt.append(single_gt)

        tgt_dir = os.path.join(self.dst_path, 'scripts')
        os.makedirs(tgt_dir, exist_ok=True)
        self.create_multi_proc_script(cmds=extract_frame_commands,
                                      tgt_dir=tgt_dir,
                                      cmd_name='extract_frame')
        self.create_multi_proc_script(cmds=extract_audio_commands,
                                      tgt_dir=tgt_dir,
                                      cmd_name='extract_audio')

        return video_gt

    def identify_video_audio_streams_from_raw_files(self):
        # get yt video id and extension from raw file names
        raw_file_rgx = re.compile('([^.]*).([^.]*).([^.]*)')
        video_files = os.listdir(self.raw_video_path)
        files_per_video_id = {}
        for file in tqdm(video_files):
            m = raw_file_rgx.match(file)
            if m:
                video_id = m.group(1)
                if not video_id in files_per_video_id:
                    files_per_video_id[video_id] = []
                files_per_video_id[video_id].append(file)
        streams_per_video_id = {}
        for video_id, files in tqdm(files_per_video_id.items()):

            stream_types = {}
            for sample in files:
                result = self.ffprobe('/bin/ffprobe', os.path.join(self.raw_video_path, sample))

                for stream in result.findall("./streams/stream"):
                    codec_type = stream.attrib['codec_type']
                    if not codec_type in stream_types:
                        stream_types[codec_type] = []
                    stream_types[codec_type].append(sample)

            streams_per_video_id[video_id] = stream_types
        return streams_per_video_id

    def create_frame_audio_extract_scripts(self):
        return self.video2frame_audio()

    def create_sentences_gt(self):
        vid_to_sentences_dict = {}
        for sent in self.vtt_gt:
            video_id = sent['videoID']
            if not video_id in vid_to_sentences_dict:
                vid_to_sentences_dict[video_id] = sent

        new_gt = []
        for single_gt in self.video_gt:
            if 'enCap' in vid_to_sentences_dict[single_gt['src_id']]:
                for cap in vid_to_sentences_dict[single_gt['src_id']]['enCap']:
                    new_gt.append(self.get_caption_gt(single_gt, cap))
            else:
                new_gt.append(self.get_caption_gt(single_gt, 'N/A'))


        return new_gt

class PrepareMSVDDData(PrepareVTTData):
    def __init__(self,
                 vtt_datasets_base_path,
                 datasplit,
                 raw_video_path,
                 raw_captions_dir,
                 raw_video_audio_path,
                 gt_filename="video_corpus.csv",
                 constant_fps=None):
        super().__init__(vtt_datasets_base_path, datasplit, raw_video_path, raw_captions_dir)
        self.gt_filename = gt_filename
        self.constant_fps = constant_fps
        self.raw_video_audio_path = raw_video_audio_path

    def ffprobe(self, executable, filename):
        '''Runs ``ffprobe`` executable over ``filename``, returns parsed XML

        Parameters:

            executable (str): Full path leading to ``ffprobe``
            filename (str): Full path leading to the file to be probed

        Returns:

            xml.etree.ElementTree: containing all parsed elements

        '''

        cmd = [
            executable,
            '-v', 'quiet',
            '-print_format', 'xml',  # here is the trick
            '-show_format',
            '-show_streams',
            filename,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE)

        # print(result.stdout.decode('utf-8'))
        return ElementTree.fromstring(result.stdout.decode('utf-8'))

    def video2frame_audio(self):

        captions_path = os.path.join(self.raw_captions_dir, self.gt_filename)

        df = pd.read_csv(captions_path)
        english_corpus = df[df['Language'] == 'English']


        video_files = os.listdir(self.raw_video_path)
        streams_per_video_id = self.identify_video_audio_streams_from_raw_files()

        not_found = []


        # split video ids and start/end time
        rgx = re.compile("(.*)_(.*)_(.*)")

        msvdd_new = []
        for xx in tqdm(english_corpus.iterrows()):
            sample = xx[1]
            raw_vid = sample['VideoID']
            # m = rgx.match(raw_vid)
            # if m:
            new_sample = {}
            new_sample['startTime'] = int(sample['Start'])
            new_sample['endTime'] = int(sample['End'])
            new_sample['Description'] = sample['Description']
            vid = f"{raw_vid}_{new_sample['startTime']}_{new_sample['endTime']}"
            new_sample['ytVideoID'] = raw_vid
            new_sample['videoID'] = vid

            if vid in streams_per_video_id:
                video_audio_files = streams_per_video_id[vid]
                # print(video_audio_files)
                if 'video' in video_audio_files:
                    correct_video_path_from_multiple_files = [x for x in video_audio_files['video'] if x[1] == vid][0][0]
                    new_sample['raw_video_path'] = os.path.join(self.raw_video_path, correct_video_path_from_multiple_files)
                else:
                    new_sample['raw_video_path'] = None

                if 'audio' in video_audio_files:
                    new_sample['raw_audio_path'] = os.path.join(self.raw_video_audio_path, video_audio_files['audio'][0][0])
                else:
                    new_sample['raw_audio_path'] = None
            else:
                new_sample['raw_audio_path'] = None
                new_sample['raw_video_path'] = None
            msvdd_new.append(new_sample)
                # Old code....
                # rexp = f".*{re.escape(vid)}.*mp4"
                # rgx3 = re.compile(rexp)
                # results = list(filter(rgx3.match, video_files))
                # if len(results) == 1:
                #     new_sample['raw_video_path'] = os.path.join(self.raw_video_path, results[0])
                # else:
                #     rexp = f".*{re.escape(vid)}.*webm"
                #     rgx2 = re.compile(rexp)
                #     results = list(filter(rgx2.match, video_files))
                #     if len(results) == 1:
                #         new_sample['raw_video_path'] = os.path.join(self.raw_video_path, results[0])
                #     else:
                #         new_sample['raw_video_path'] = None
                #         not_found.append(new_sample['videoID'])
                #
                # vatex_new.append(new_sample)

        self.vtt_gt = msvdd_new

        frames_path = os.path.join(self.dst_path, 'frames')
        audio_path = os.path.join(self.dst_path, 'audio')

        extract_frame_commands = [f"mkdir -p {frames_path}"]
        extract_audio_commands = [f"mkdir -p {audio_path}"]

        video_gt = []
        vid_in_cmds = set()

        for sample in self.vtt_gt:
            if sample['raw_video_path'] != None:
                video = sample['raw_video_path']
                audio = sample['raw_audio_path']

                video_id = f"video_{self.datasplit}_{sample['videoID']}"

                frames_dst_path = os.path.join(frames_path, video_id)
                mkdir_frame_cmd = f"mkdir {frames_dst_path}"

                if self.constant_fps is not None:
                    fpscmd = f' -vf fps={self.constant_fps}'
                else:
                    fpscmd = ''

                ffmpeg_cmd = f"ffmpeg -i {video}{fpscmd} '{frames_dst_path}/%d.jpg'"
                ffmpeg_audio_cmd = f"ffmpeg -i {audio} -ss {sample['startTime']} -t {sample['endTime'] - sample['startTime']} -q:a 0 '{audio_path}/{video_id}.wav'"

                # ffmpeg_cmd = f"ffmpeg -i {video} '{frames_dst_path}/%d.jpg'"
                # ffmpeg_audio_cmd = f"ffmpeg -i {video} '{audio_path}/{video_id}.wav'"
                if sample['videoID'] not in vid_in_cmds:
                    extract_frame_commands.append(" && ".join([mkdir_frame_cmd, ffmpeg_cmd]))  # , rm_cmd, rm_dir ]))
                    extract_audio_commands.append(ffmpeg_audio_cmd)  # , rm_cmd, rm_dir ]))
                    vid_in_cmds.add(sample['videoID'])

                    single_gt = {
                        'raw_video_path': video,
                        'video_id': video_id,
                        'src_id': sample['ytVideoID'],
                        'video_id_with_timestamps': sample['videoID'],
                    }
                    video_gt.append(single_gt)

        tgt_dir = os.path.join(self.dst_path, 'scripts')
        os.makedirs(tgt_dir, exist_ok=True)
        self.create_multi_proc_script(cmds=extract_frame_commands,
                                      tgt_dir=tgt_dir,
                                      cmd_name='extract_frame')
        self.create_multi_proc_script(cmds=extract_audio_commands,
                                      tgt_dir=tgt_dir,
                                      cmd_name='extract_audio')

        return video_gt

    def identify_video_audio_streams_from_raw_files(self):
        # get yt video id and extension from raw file names
        raw_file_rgx = re.compile('([^.]*)_([^.]*)_([^.]*)')
        raw_file_rgx_self_downloaded = re.compile('([^.]*).([^.]*).([^.]*)')
        video_files = os.listdir(self.raw_video_path)
        files_per_video_id = {}
        x = 0
        for file in tqdm(video_files):
            m = raw_file_rgx.match(file)
            if m:
                video_id = m.group(1)
                if not video_id in files_per_video_id:
                    files_per_video_id[video_id] = []
                files_per_video_id[video_id].append((file, f"{m.group(1)}_{m.group(2)}_{m.group(3)}"))
            else:
                print(file)

        audio_files = os.listdir(self.raw_video_audio_path)
        audio_files_per_video_id = {}
        for file in tqdm(audio_files):
            m2 = raw_file_rgx_self_downloaded.match(file)
            if m2:
                video_id = m2.group(1)
                if not video_id in audio_files_per_video_id:
                    audio_files_per_video_id[video_id] = []
                audio_files_per_video_id[video_id].append((file, f"{m2.group(1)}_{m2.group(2)}_{m2.group(3)}"))

        streams_per_video_id = {}
        for video_id, files_id_tuple in tqdm(files_per_video_id.items()):
            stream_types = {}
            for tuple_sample in files_id_tuple:
                files, full_id = tuple_sample
                result = self.ffprobe('/bin/ffprobe', os.path.join(self.raw_video_path, files))

                for stream in result.findall("./streams/stream"):
                    codec_type = stream.attrib['codec_type']
                    if not codec_type in stream_types:
                        stream_types[codec_type] = []
                    stream_types[codec_type].append(tuple_sample)

                streams_per_video_id[full_id] = {'video_id': video_id}
                streams_per_video_id[full_id].update(stream_types)

                if video_id in audio_files_per_video_id:
                    files_id_aud_tuple = audio_files_per_video_id[video_id]

                    stream_types_aud = {}
                    for tuple_sample_aud in files_id_aud_tuple:
                        files_aud, full_id_aud = tuple_sample_aud
                        result = self.ffprobe('/bin/ffprobe',
                                              os.path.join(self.raw_video_audio_path, files_aud))

                        for stream in result.findall("./streams/stream"):
                            codec_type = stream.attrib['codec_type']
                            if not codec_type in stream_types_aud:
                                stream_types_aud[codec_type] = []
                            stream_types_aud[codec_type].append(tuple_sample_aud)

                        if full_id in streams_per_video_id:
                            if 'audio' in stream_types_aud:
                                streams_per_video_id[full_id]['audio'] = stream_types_aud['audio']

        return streams_per_video_id

    def create_frame_audio_extract_scripts(self):
        return self.video2frame_audio()

    def create_sentences_gt(self):
        vid_to_sentences_dict = {}
        for sent in self.vtt_gt:
            if not type(sent['Description']) is str and math.isnan(sent['Description']):
                continue
            video_id = sent['videoID']
            if not video_id in vid_to_sentences_dict:
                vid_to_sentences_dict[video_id] = [sent]
            else:
                vid_to_sentences_dict[video_id].append(sent)

        new_gt = []

        for video_id_with_timestamps, caps in vid_to_sentences_dict.items():
            for cap in caps:
                single_gt = cap.copy()
                single_gt['video_id'] = cap['videoID']
                new_gt.append(self.get_caption_gt(single_gt, cap['Description']))


        return new_gt



ptd = PrepareMSVDDData(vtt_datasets_base_path='/path/to/Datasets/',
                         datasplit='MSVDD',
                         raw_video_path='/path/to/Datasets/MSVD_raw/YouTubeClips/',
                       raw_video_audio_path='/path/to/Datasets/MSVD_raw/YouTubeClips_download/videos',
                         raw_captions_dir='/path/to/Datasets/MSVD_raw/')

ptd.create_dataset_info()