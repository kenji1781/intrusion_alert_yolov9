import cv2
import torch
import numpy as np
import json
import argparse
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, config_file='config.json', confidence_threshold=0.5):
        """
        人物検出システムの初期化
        
        Args:
            config_file (str): 設定ファイルのパス
            confidence_threshold (float): 検出の信頼度閾値
        """
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # YOLOv9モデルをロード
        try:
            print("YOLOモデルを読み込み中...")
            # YOLOv9の事前訓練済みモデルを使用
            # 利用可能なモデル: yolov9n.pt, yolov9s.pt, yolov9m.pt, yolov9l.pt, yolov9x.pt
            self.model = YOLO('yolov9n.pt')
            #self.model = YOLO('yolov8n-pose.pt')  
              
            print(f"モデルをデバイス {self.device} で実行します")
            self.use_yolo = True
        except Exception as e:
            print(f"YOLOの読み込みに失敗しました: {e}")
            print("YOLOにフォールバック...")
            try:
                self.model = YOLO('yolov9s.pt')
                print("YOLOモデルを使用します")
                self.use_yolo = True
            except Exception as e2:
                print(f"YOLOの読み込みも失敗しました: {e2}")
                print("代替として、OpenCVのHaarcascadeを使用します...")
                self.use_yolo = False
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 設定ファイルを読み込み
        self.load_config(config_file)

   

    def load_config(self, config_file):
        """設定ファイルを読み込み"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 検出エリアの座標 (x, y, width, height)
            self.detection_areas = config.get('detection_areas', [])
            
            # カメラ設定
            self.camera_index = config.get('camera_index', 0)
            self.frame_width = config.get('frame_width', 640)
            self.frame_height = config.get('frame_height', 480)
            
            print(f"設定ファイル '{config_file}' を読み込みました")
            print(f"検出エリア数: {len(self.detection_areas)}")
            
        except FileNotFoundError:
            print(f"設定ファイル '{config_file}' が見つかりません。デフォルト設定を使用します。")
            self.create_default_config(config_file)
            
    def create_default_config(self, config_file):
        """デフォルト設定ファイルを作成"""
        default_config = {
            "camera_index": 0,
            "frame_width": 640,
            "frame_height": 480,
            "detection_areas": [
                {
                    "name": "Area1",
                    "x": 100,
                    "y": 100,
                    "width": 200,
                    "height": 200
                },
                {
                    "name": "Area2", 
                    "x": 350,
                    "y": 150,
                    "width": 180,
                    "height": 180
                }
            ]
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        self.detection_areas = default_config['detection_areas']
        self.camera_index = default_config['camera_index']
        self.frame_width = default_config['frame_width']
        self.frame_height = default_config['frame_height']
        
        print(f"デフォルト設定ファイル '{config_file}' を作成しました")
    
    def detect_persons_yolo(self, frame):
        """YOLOを使用した人物検出"""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # クラス0は人物（COCO dataset）
                    if int(box.cls) == 0 and float(box.conf) >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        detections.append([x1, y1, x2, y2, conf, 0])
        
        return detections
    
    def detect_persons_opencv(self, frame):
        """OpenCVのHaarcascadeを使用した顔検出（代替手段）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            # 顔検出結果を人物検出形式に変換
            detections.append([x, y, x + w, y + h, 0.8, 0])  # 信頼度は固定値
        
        return detections
        
    def is_person_in_area(self, detections, area):
        """指定エリア内に人物がいるかチェック"""
        area_x1 = area['x']
        area_y1 = area['y']
        area_x2 = area_x1 + area['width']
        area_y2 = area_y1 + area['height']
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # バウンディングボックスの中心点を計算
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 中心点がエリア内にあるかチェック
            if (area_x1 <= center_x <= area_x2 and 
                area_y1 <= center_y <= area_y2):
                return True
                
            # または、バウンディングボックスとエリアが重複しているかチェック
            if not (x2 < area_x1 or x1 > area_x2 or y2 < area_y1 or y1 > area_y2):
                return True
        
        return False
        
    def draw_detection_areas(self, frame, detections):
        """検出エリアを描画"""
        person_detected_any = False
        
        for area in self.detection_areas:
            x, y, w, h = area['x'], area['y'], area['width'], area['height']
            name = area['name']
            
            # エリア内に人物がいるかチェック
            person_detected = self.is_person_in_area(detections, area)
            if person_detected:
                person_detected_any = True
            
            # 色を設定（人物検出時は赤、そうでなければ緑）
            color = (0, 0, 255) if person_detected else (0, 255, 0)  # BGR
            thickness = 3 if person_detected else 2
            
            # エリアの枠を描画
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # エリア名とステータスを表示
            status = "DETECTED!" if person_detected else "MONITORING"
            label = f"{name}: {status}"
            
            # テキストのサイズを取得
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # テキスト背景を描画
            cv2.rectangle(frame, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # テキストを描画
            cv2.putText(frame, label, (x, y - 5), font, font_scale, 
                       (255, 255, 255), text_thickness)
        
        return frame, person_detected_any
        
    def draw_person_detections(self, frame, detections):
        """検出された人物にバウンディングボックスを描画"""
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # バウンディングボックスを描画
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (255, 255, 0), 2)  # 黄色
            
            # 信頼度を表示
            detection_type = "Person" if self.use_yolo else "Face"
            label = f"{detection_type}: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
        
    def run(self):
        """メイン実行ループ"""
        # カメラを初期化
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"カメラ（インデックス: {self.camera_index}）を開けませんでした")
            return
        
        # フレームサイズを設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # 使用中のモデルを表示
        if self.use_yolo:
            model_name = self.model.model_name if hasattr(self.model, 'model_name') else "YOLO"
            detection_method = f"{model_name}"
        else:
            detection_method = "OpenCV Face Detection"
            
        print(f"Person Detection System Started (Method: {detection_method})")
        print("Press 'q' to quit")
        print("Press 'r' to reload config")

        cv2.namedWindow('PersonDetectionSystem', cv2.WINDOW_NORMAL)
        
        y = 0  # 初期状態のPLC通信値

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("フレームの取得に失敗しました")
                    break
                
                # 人物検出を実行
                if self.use_yolo:
                    detections = self.detect_persons_yolo(frame)
                else:
                    detections = self.detect_persons_opencv(frame)
                
                # 検出エリアを描画
                frame, person_detected = self.draw_detection_areas(frame, detections)

                #                 # PLC通信を行う
                # x = 1 if person_detected else 0
                # if x != y:
                #     if person_detected:
                #         # PLCにデータを送信
                #         r = self.plc_com(add='DM0.U', data='1')
                #     else:
                #         r = self.plc_com(add='DM0.U', data='0')
                # y = x
                
                
                # 人物検出結果を描画
                frame = self.draw_person_detections(frame, detections)
                
                # フレーム情報を表示
                info_text = f"Detection Areas: {len(self.detection_areas)} | Method: {detection_method}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                # 検出状態を表示
                status_text = f"Status: {'PERSON DETECTED' if person_detected else 'MONITORING'}"
                status_color = (0, 0, 255) if person_detected else (0, 255, 0)
                cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, status_color, 2)
                
                # フレームを拡大表示
                cv2.imshow('PersonDetectionSystem', cv2.resize(frame, (1280, 960)))
                
                # キー入力をチェック
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Reloading configuration...")
                    self.load_config('config.json')
                    
        except KeyboardInterrupt:
            print("\n終了しています...")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='リアルタイム人物検出システム (YOLO対応)')
    parser.add_argument('--config', '-c', default='config.json', 
                       help='設定ファイルのパス (デフォルト: config.json)')
    parser.add_argument('--confidence', '-conf', type=float, default=0.5,
                       help='検出の信頼度閾値 (デフォルト: 0.5)')
    parser.add_argument('--model', '-m', default='yolov9s.pt',
    choices=['yolov9-c.pt', 'yolov9s.pt', 'yolov9m.pt', 'yolov9l.pt', 'yolov9x.pt', 'yolov9-c.pt'],
    help='使用するYOLOモデル (デフォルト: yolov9s.pt)')
    
    args = parser.parse_args()
    
    # 人物検出システムを初期化して実行
    detector = PersonDetector(args.config, args.confidence)
    # モデルを指定されたものに変更
    if detector.use_yolo:
        try:
            detector.model = YOLO(args.model)
            print(f"モデルを {args.model} に変更しました")
        except Exception as e:
            print(f"指定されたモデル {args.model} の読み込みに失敗: {e}")
            print("デフォルトモデルを使用します")
    
    detector.run()

if __name__ == "__main__":
    main()