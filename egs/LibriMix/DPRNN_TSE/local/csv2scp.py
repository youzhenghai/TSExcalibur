import argparse
import pandas as pd
import os

#/work/youzhenghai/project/speakerbeam/egs/libri2mix/data/wav16k/min/test/mixture_test_mix_clean.csv
#/work/youzhenghai/project/speaker_extraction_SpEx/data/test/aux.scp


#python lib2scp.py /work/youzhenghai/project/tdspeakerbeam/egs/libri2mix/data/wav16k/min/train-100/mixture_train-100_mix_clean.csv /work/youzhenghai/project/speaker_extraction_SpEx_archieve/data_new/train-100/mix.scp /work/youzhenghai/project/speaker_extraction_SpEx_archieve/data_new/train-100/ref.scp /work/youzhenghai/project/speaker_extraction_SpEx_archieve/data_new/train-100/spk.scp

def create_scp_files(csv_file, mix_scp_file, ref_scp_file, spk_file):
    df = pd.read_csv(csv_file)
    unique_speakers = set()

    # 创建mix.scp文件
    with open(mix_scp_file, 'w') as mix_scp_file:
        for index, row in df.iterrows():
            mixture_id = row['mixture_ID']
            mixture_path = row['mixture_path']

            speakerA_contA_textA, speakerB_contB_textB = mixture_id.split('_')
            speakerA, contA, textA = speakerA_contA_textA.split('-')
            speakerB, contB, textB = speakerB_contB_textB.split('-')
            
            # .spk 文件保存
            if speakerA:
                unique_speakers.add(speakerA)
            if speakerB:
                unique_speakers.add(speakerB)

            for i in [speakerA_contA_textA,speakerB_contB_textB]: 
                mix_scp_file.write(f'{mixture_id}_{i} {mixture_path}\n')


    with open(ref_scp_file, 'w') as ref_scp_file:
        for index, row in df.iterrows():
            mixture_id = row['mixture_ID']

            speakerA_contA_textA, speakerB_contB_textB = mixture_id.split('_')

            source_1_path = row['source_1_path']
            source_2_path = row['source_2_path']
            idcount = 0
            for i in [speakerA_contA_textA,speakerB_contB_textB]:  
                ref_scp_file.write(f'{mixture_id}_{i} {source_1_path if idcount == 0 else source_2_path}\n')
                idcount = idcount + 1


    if os.path.exists(spk_file):
        with open(spk_file, 'r') as existing_spk_file:
            existing_speakers = existing_spk_file.read().splitlines()
            unique_speakers.update(existing_speakers)
    with open(spk_file, 'w') as spk_file:
        for speaker in unique_speakers:
            spk_file.write(f'{speaker}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mix.scp, ref.scp, and spk_file from a CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("mix_scp_file", help="Path to the mix.scp output file")
    parser.add_argument("ref_scp_file", help="Path to the ref.scp output file")
    parser.add_argument("spk_file", help="Path to the spk_file output file")

    args = parser.parse_args()
    
    create_scp_files(args.csv_file, args.mix_scp_file, args.ref_scp_file, args.spk_file)
