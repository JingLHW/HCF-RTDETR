import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
from datetime import datetime
import sys
import pandas as pd

from ultralytics import RTDETR, YOLO


if __name__ == '__main__':
    # ======================
    # 初始化模型
    # ======================
    model = RTDETR('/root/autodl-tmp/code/rtdetr-r18.yaml')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = 'rtdetr/train_rtdetr-r18/exp_rtdetr-r18'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.txt")

    sys.stdout = open(log_path, 'w', buffering=1)
    print(f"📝 Training started at {timestamp}")
    print(f"Logging to: {log_path}")

    # ======================
    # 启动训练
    # ======================
    results = model.train(
        data='/root/autodl-tmp/data/data_xin.yaml',
        cache=False,
        imgsz=640,
        epochs=150,
        batch=16,
        workers=4,
        device='0',
        project='rtdetr-r18/train',
        name='exp_rtdetr-r18',
        warmup_epochs=5


    )

    # ======================
    # 读取 best 指标
    # ======================
    try:
        csv_path = os.path.join(results.save_dir, "results.csv")
        df = pd.read_csv(csv_path)
        best_idx = df['metrics/mAP50-95(B)'].idxmax()
        best_epoch = int(df.loc[best_idx, 'epoch'])
        best_map = float(df.loc[best_idx, 'metrics/mAP50-95(B)'])

        print("\n==========================")
        print(f"🏆 Best epoch: {best_epoch}")
        print(f"🏆 Best mAP@0.5:0.95 = {best_map:.4f}")
        print("==========================")
    except Exception as e:
        print("⚠️ Could not parse best metrics:", e)


    sys.stdout.close()