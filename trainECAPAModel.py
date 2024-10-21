import argparse
import glob
import os
import torch
import warnings
import time
import re  # 정규 표현식 사용을 위한 모듈
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): 성능이 개선되지 않아도 허용할 최대 에포크 수
            min_delta (float): 성능이 개선되었다고 간주할 최소한의 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def check_file_paths(eval_list):
    """
    eval_list 파일에 있는 경로들이 실제로 존재하는지 확인하는 함수
    """
    print("Checking evaluation file paths...")

    try:
        with open(eval_list, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print("UTF-8 인코딩 실패, cp949로 다시 시도합니다.")
        with open(eval_list, 'r', encoding='cp949') as f:
            lines = f.readlines()

    # 정규식을 사용하여 경로 추출
    for line in lines:
        match = re.match(r'(\d)\s+(.*?\.wav)\s+(.*?\.wav)', line.strip())
        if not match:
            print(f"잘못된 형식의 줄: {line.strip()}")
            continue

        label = match.group(1)
        path1 = match.group(2).strip('"')  # 인용 부호 제거
        path2 = match.group(3).strip('"')  # 인용 부호 제거

        if not os.path.exists(path1):
            print(f"Error: 파일 1이 존재하지 않습니다: {path1}")
        if not os.path.exists(path2):
            print(f"Error: 파일 2가 존재하지 않습니다: {path2}")


def main():
    parser = argparse.ArgumentParser(description="ECAPA_trainer")

    ## Training Settings
    parser.add_argument('--num_frames', type=int, default=200, help='Duration of the input segments, e.g., 200 for 2 seconds')
    parser.add_argument('--max_epoch', type=int, default=80, help='Maximum number of epochs')  # 에포크 최대값을 더 크게 설정
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (number of epochs without improvement)')

    ## Paths
    parser.add_argument('--train_list', type=str, default="C:\\Users\\sjiye\\Documents\\continuous01-train.txt", help='The path of the training list')
    parser.add_argument('--train_path', type=str, default="C:\\Users\\sjiye\\Documents\\003\\01.데이터\\1.Training\\원천데이터\\TS_continuous_01\\continuous", help='The path of the training data')
    parser.add_argument('--eval_list', type=str, default="C:\\Users\\sjiye\\Documents\\continuous_eval_list.txt", help='The path of the evaluation list')
    parser.add_argument('--eval_path', type=str, default="C:\\Users\\sjiye\\Documents\\003\\01.데이터\\2.Validation\\원천데이터\\VS_continuous_01\\continuous", help='The path of the evaluation data')
    parser.add_argument('--musan_path', type=str, default="/data08/Others/musan_split", help='The path to the MUSAN set')
    parser.add_argument('--rir_path', type=str, default="/data08/Others/RIRS_NOISES/simulated_rirs", help='The path to the RIR set')
    parser.add_argument('--save_path', type=str, default='C:\\Users\\sjiye\\Documents', help='Path to save the score.txt and models')
    parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')
    parser.add_argument('--score_save_path', type=str, default="score.txt", help='Path to save the score.txt')

    ## Model and Loss settings
    parser.add_argument('--C', type=int, default=512, help='Channel size for the speaker encoder')
    parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
    parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
    parser.add_argument('--n_class', type=int, default=1863, help='Number of speakers')

    ## Command
    parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')

    ## Initialization
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    args = init_args(args)

    ## Define the data loader
    trainloader = train_loader(**vars(args))
    trainLoader = torch.utils.data.DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)

    ## 데이터 로드 상태 확인
    print("Training data successfully loaded!")
    print(f"Number of batches: {len(trainLoader)}")
    print(f"Batch size: {args.batch_size}")

    ## 파일 경로 확인 (eval_list와 eval_path 확인)
    check_file_paths(args.eval_list)

    ## 모델 초기화 및 파라미터 로드
    modelfiles = glob.glob(os.path.join(args.save_path, 'model_0*.model'))
    modelfiles.sort()

    ## Early Stopping 설정
    early_stopping = EarlyStopping(patience=args.patience)

    ## 모델 불러오기 (initial_model이 있으면 이어서 학습)
    if args.initial_model:
        print(f"Model {args.initial_model} loaded from previous state!")
        s = ECAPAModel(**vars(args))
        s.load_parameters(args.initial_model)
        epoch = int(args.initial_model.split('_')[-1].split('.')[0]) + 1  # 기존 에포크에서 이어서 시작
    else:
        epoch = 1
        s = ECAPAModel(**vars(args))

    EERs = []
    score_file = open(args.score_save_path, "a+")

    ## 학습 루프
    while True:
        ## Training for one epoch
        print(f"Epoch {epoch} starting...")
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

        # 정확도 출력 시 텐서에서 숫자로 변환
        print(f"Epoch {epoch} completed: Loss: {loss:.4f}, Learning Rate: {lr:.6f}, Accuracy: {acc.item():.2f}%")

        ## Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            model_save_path = os.path.join(args.save_path, f"model_{epoch:04d}.model")
            s.save_parameters(model_save_path)
            EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path)
            EERs.append(EER)
            bestEER = min(EERs)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {epoch} epoch, ACC {acc.item():.2f}%, EER {EER:.2f}%, bestEER {bestEER:.2f}%")
            score_file.write(f"{epoch} epoch, LR {lr:.6f}, LOSS {loss:.4f}, ACC {acc.item():.2f}%, EER {EER:.2f}%, bestEER {bestEER:.2f}%\n")
            score_file.flush()

            ## Early Stopping 체크
            early_stopping(EER)
            if early_stopping.early_stop:
                print("Early stopping activated.")
                break

        if epoch >= args.max_epoch:
            print("Training complete!")
            break

        epoch += 1

    score_file.close()


if __name__ == "__main__":
    main()
