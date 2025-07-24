import cv2
import time

# 最大で試行するカメラのインデックス番号
max_cameras = 50 # 必要に応じてこの数を増やす

print("利用可能なカメラインデックスを検索中...")
print("各カメラの映像が短時間表示されます。")
print("目的のWebカメラの映像が表示されたときのインデックス番号をメモしてください。")
print("-" * 40)

found_cameras = {}

for i in range(max_cameras):
    print(f"カメラインデックス {i} をテスト中...")
    cap = cv2.VideoCapture(i)
    
    if not cap.isOpened():
        print(f"カメラインデックス {i}: 利用不可")
        continue

    # カメラが開けた場合、映像を取得して表示
    ret, frame = cap.read()
    if ret:
        print(f"カメラインデックス {i}: 利用可能 - 映像を表示します (2秒間)")
        cv2.imshow(f"Camera Index {i} (Press any key to close)", frame)
        
        # 2秒間表示するか、キーが押されたら閉じる
        key = cv2.waitKey(2000) 
        if key != -1: # 何かキーが押されたらループを終了
            cap.release()
            cv2.destroyAllWindows()
            print("\nユーザーによってテストが中断されました。")
            break

        found_cameras[i] = "利用可能"
        cv2.destroyAllWindows() # 表示ウィンドウを閉じる
    else:
        print(f"カメラインデックス {i}: 利用可能だが映像取得失敗 (内蔵カメラの可能性あり)")
    
    cap.release() # カメラを解放

print("-" * 40)
print("テストが完了しました。")
if found_cameras:
    print("検出されたカメラインデックス:")
    for index, status in found_cameras.items():
        print(f"  カメラインデックス {index}: {status}")
    print("\n目的の外付けWebカメラの映像が表示されたときのインデックス番号を、OpenCVのVideoCapture()に渡してください。")
else:
    print("利用可能なカメラが見つかりませんでした。")
    print("Webカメラの接続を確認してください。")
