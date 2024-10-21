import torch
import librosa
import numpy as np
import os
from moviepy.editor import VideoFileClip  # 동영상에서 음성 추출
from ECAPAModel import ECAPAModel  # ECAPAModel.py에서 모델 불러오기

# 모델 로드 (.model 파일 경로)
MODEL_PATH = "C:\\Users\\sjiye\\Downloads\\pretrain.model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ECAPAModel 초기화 후 파라미터 로드 (device 인자 추가)
model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=1863, m=0.2, s=30, test_step=10, device=device).to(device)

# 저장된 .model 파일로부터 가중치 로드
model.load_parameters(MODEL_PATH)
model.eval()

# 동영상에서 음성을 추출하는 함수
def extract_audio_from_video(video_path: str):
    try:
        video = VideoFileClip(video_path)
        audio_path = video_path.replace(".mp4", ".wav")  # 동영상 파일명과 동일한 이름의 .wav 파일로 저장
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)  # 경고 메시지 억제
        print(f"동영상에서 음성을 추출하여 {audio_path}에 저장했습니다.")
        return audio_path

    except Exception as e:
        print(f"동영상 음성 추출 오류: {e}")
        return None

# 오디오 파일을 모델의 입력 형식으로 변환하는 함수
def preprocess_audio(file_path: str):
    try:
        # librosa를 사용하여 오디오 파일을 로드
        y, sr = librosa.load(file_path, sr=16000)  # ECAPA 모델은 16kHz 샘플링을 기대할 수 있음
        max_audio = 300 * 160 + 240  # 3초 길이로 패딩

        if len(y) < max_audio:
            y = np.pad(y, (0, max_audio - len(y)), 'wrap')  # 짧은 오디오에 대해 패딩
        else:
            y = y[:max_audio]  # 오디오가 너무 긴 경우 자름

        # 오디오 신호를 배치 형태로 변환
        audio_tensor = torch.FloatTensor(np.stack([y], axis=0)).to(device)

        return audio_tensor

    except Exception as e:
        print(f"오디오 전처리 오류: {e}")
        return None

# 두 화자를 비교하는 함수 (추가 연산 없이 원래 코사인 유사도를 사용)
def compare_speakers(audio1, audio2):
    try:
        # 모델을 사용하여 임베딩 생성
        with torch.no_grad():
            embedding1 = model(audio1)
            embedding2 = model(audio2)

        # 코사인 유사도 계산
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)

        # 코사인 유사도 값이 0.7 이상이면 동일인으로 판단
        if similarity.item() >= 0.4:
            return "same", similarity.item()  # 동일인
        else:
            return "different", similarity.item()  # 동일인 아님

    except Exception as e:
        print(f"화자 비교 오류: {e}")
        return None, None

# 테스트를 위한 로컬 파일 경로
proper_speaker_file = "C:\\Users\\sjiye\\Downloads\\예린1.mp4"  # 적합한 화자의 동영상 파일 경로
compare_speaker_file = "C:\\Users\\sjiye\\Downloads\\지연1.mp4"  # 비교할 화자의 동영상 파일 경로

# 적합한 화자의 파일을 처리하여 오디오 추출 및 전처리
audio1_path = extract_audio_from_video(proper_speaker_file)
audio2_path = extract_audio_from_video(compare_speaker_file)

if audio1_path and audio2_path:
    audio1 = preprocess_audio(audio1_path)
    audio2 = preprocess_audio(audio2_path)

    if audio1 is not None and audio2 is not None:
        # 두 화자 비교
        comparison_result, similarity_score = compare_speakers(audio1, audio2)
        if comparison_result is not None:
            if comparison_result == "same":
                print(f"두 화자는 동일인입니다. 유사도 점수: {similarity_score:.4f}")
            else:
                print(f"두 화자는 동일인이 아닙니다. 유사도 점수: {similarity_score:.4f}")
        else:
            print("화자 비교 실패")
else:
    print("오디오 파일 처리 실패")
